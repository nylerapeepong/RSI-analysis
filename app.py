"""
Streamlit RSI/EMA statistical backtester.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit yfinance pandas pandas-ta numpy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# pandas-ta currently imports more reliably on Python 3.13 when numba JIT is
# disabled before import. Indicator calculations still run fast for daily data.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    import pandas_ta as ta
except Exception as exc:  # pragma: no cover - package imports can fail at runtime.
    ta = None
    PANDAS_TA_IMPORT_ERROR = exc
else:
    PANDAS_TA_IMPORT_ERROR = None


RSI_LENGTH = 14
EMA_LENGTH = 200
TARGET_RETURN = 0.10
COOLDOWN_CANDLES = 14
HOLDING_PERIODS = (3, 7, 14, 28)
YEARS_TO_FETCH = "5y"
RSI_TRIGGER_LEVEL = 30
HISTORY_OPTIONS = ("1y", "2y", "3y", "5y", "10y", "max")


@dataclass(frozen=True)
class SignalGroup:
    """Configuration for one trigger family."""

    key: str
    label: str
    signal_column: str


@dataclass(frozen=True)
class EmaSegment:
    """Configuration for one EMA segment."""

    key: str
    label: str


@dataclass(frozen=True)
class BacktestConfig:
    """User-adjustable backtest settings."""

    rsi_length: int
    ema_length: int
    target_return: float
    cooldown_candles: int
    history_period: str
    holding_periods: tuple[int, ...]


SIGNAL_GROUPS = (
    SignalGroup(
        key="cross_down",
        label="RSI Cross Down 30",
        signal_column="rsi_cross_down_30",
    ),
    SignalGroup(
        key="cross_up",
        label="RSI Cross Up 30",
        signal_column="rsi_cross_up_30",
    ),
)

EMA_SEGMENTS = (
    EmaSegment(key="above_ema", label="Above EMA (Buy the Dip)"),
    EmaSegment(key="below_ema", label="Below EMA (Falling Knife)"),
    EmaSegment(key="all", label="All Signals Combined"),
)


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output into a simple OHLCV dataframe.

    yfinance can return multi-index columns for some inputs. This function keeps
    the script resilient for both ordinary tickers and edge cases.
    """

    if data.empty:
        return data

    normalized = data.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {', '.join(missing)}")

    normalized = normalized.loc[:, required_columns].dropna(subset=["High", "Low", "Close"])
    normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price_history(ticker: str, history_period: str) -> pd.DataFrame:
    """Download daily historical data for a ticker and history period."""

    raw = yf.download(
        ticker,
        period=history_period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_ohlcv(raw)


def add_indicators(data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Add RSI, EMA, and raw RSI threshold crosses."""

    enriched = data.copy()
    if ta is not None:
        enriched["RSI"] = ta.rsi(enriched["Close"], length=config.rsi_length)
        enriched["EMA"] = ta.ema(enriched["Close"], length=config.ema_length)
    else:
        enriched["RSI"] = calculate_rsi(enriched["Close"], length=config.rsi_length)
        enriched["EMA"] = enriched["Close"].ewm(
            span=config.ema_length,
            adjust=False,
            min_periods=config.ema_length,
        ).mean()

    previous_rsi = enriched["RSI"].shift(1)
    enriched["rsi_cross_down_30"] = (
        (previous_rsi >= RSI_TRIGGER_LEVEL) & (enriched["RSI"] < RSI_TRIGGER_LEVEL)
    )
    enriched["rsi_cross_up_30"] = (
        (previous_rsi <= RSI_TRIGGER_LEVEL) & (enriched["RSI"] > RSI_TRIGGER_LEVEL)
    )
    return enriched.dropna(subset=["RSI", "EMA"])


def calculate_rsi(close: pd.Series, length: int) -> pd.Series:
    """
    Calculate Wilder RSI without external indicator dependencies.

    pandas-ta is preferred above. This fallback keeps the app usable in Python
    environments where pandas-ta imports fail because of compiled dependencies.
    """

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    average_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(100)


def apply_cooldown(signal_mask: pd.Series, cooldown_candles: int) -> pd.Series:
    """
    Keep the first signal, then suppress any additional signals for N candles.

    Example: with a 14-candle cooldown, if a signal appears today, the next
    eligible signal may only occur after 14 subsequent candles have passed.
    """

    filtered = pd.Series(False, index=signal_mask.index)
    cooldown_until_position = -1

    for position, has_signal in enumerate(signal_mask.fillna(False).to_numpy()):
        if has_signal and position > cooldown_until_position:
            filtered.iat[position] = True
            cooldown_until_position = position + cooldown_candles

    return filtered


def segment_mask(data: pd.DataFrame, segment_key: str) -> pd.Series:
    """Return the boolean mask for one EMA segment."""

    if segment_key == "above_ema":
        return data["Close"] > data["EMA"]
    if segment_key == "below_ema":
        return data["Close"] < data["EMA"]
    if segment_key == "all":
        return pd.Series(True, index=data.index)
    raise ValueError(f"Unknown segment: {segment_key}")


def iter_forward_windows(
    data: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    holding_period: int,
) -> Iterable[tuple[pd.Timestamp, pd.Series, pd.Series, float]]:
    """
    Yield forward high/low windows after each signal.

    The trigger day's close is the entry price. Forward windows start on the next
    candle because the target is evaluated after the signal day has closed.
    """

    for signal_date in signal_dates:
        signal_position = data.index.get_loc(signal_date)
        entry_close = float(data.at[signal_date, "Close"])
        window = data.iloc[signal_position + 1 : signal_position + 1 + holding_period]

        if len(window) < holding_period:
            continue

        yield signal_date, window["High"], window["Low"], entry_close


def evaluate_signal_set(
    data: pd.DataFrame,
    signal_dates: pd.Index,
    holding_period: int,
    target_return: float,
) -> dict[str, float | int]:
    """Calculate one row of metrics for a signal set and holding period."""

    target_hits: list[bool] = []
    drawdowns: list[float] = []
    days_to_target: list[int] = []

    for _, forward_highs, forward_lows, entry_close in iter_forward_windows(
        data, signal_dates, holding_period
    ):
        target_price = entry_close * (1 + target_return)
        hit_positions = np.flatnonzero(forward_highs.to_numpy() >= target_price)
        hit_target = len(hit_positions) > 0

        target_hits.append(hit_target)
        if hit_target:
            target_day_position = int(hit_positions[0])
            days_to_target.append(target_day_position + 1)
            drawdown_window = forward_lows.iloc[: target_day_position + 1]
        else:
            drawdown_window = forward_lows

        lowest_low = float(drawdown_window.min())
        drawdowns.append((lowest_low / entry_close - 1) * 100)

    total_rounds = len(target_hits)
    win_rate = float(np.mean(target_hits) * 100) if total_rounds else np.nan
    average_drawdown = float(np.mean(drawdowns)) if total_rounds else np.nan
    median_days = float(np.median(days_to_target)) if days_to_target else np.nan

    return {
        "Holding Period": f"{holding_period} Days",
        "Total Rounds": total_rounds,
        "Win Rate (%)": win_rate,
        "Loss Rate (%)": 100 - win_rate if total_rounds else np.nan,
        "Avg. Maximum Drawdown (%)": average_drawdown,
        "Median Days to Target": median_days,
    }


def build_signal_details(
    data: pd.DataFrame,
    signal_dates: pd.Index,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Build per-signal rows showing exactly when each target was reached."""

    rows: list[dict[str, object]] = []

    for signal_date in signal_dates:
        signal_position = data.index.get_loc(signal_date)
        entry_close = float(data.at[signal_date, "Close"])
        target_price = entry_close * (1 + config.target_return)

        row: dict[str, object] = {
            "Signal Date": signal_date.date(),
            "Entry Close": entry_close,
            "Target Price": target_price,
            "RSI": float(data.at[signal_date, "RSI"]),
            "EMA": float(data.at[signal_date, "EMA"]),
            "Trend vs EMA": (
                "Above EMA"
                if data.at[signal_date, "Close"] > data.at[signal_date, "EMA"]
                else "Below EMA"
            ),
        }

        for holding_period in config.holding_periods:
            window = data.iloc[signal_position + 1 : signal_position + 1 + holding_period]
            suffix = f"{holding_period}D"

            if len(window) < holding_period:
                row[f"Hit Target {suffix}"] = "Incomplete"
                row[f"Days to Target {suffix}"] = np.nan
                row[f"Max High {suffix} (%)"] = np.nan
                row[f"Max Drawdown {suffix} (%)"] = np.nan
                continue

            hit_positions = np.flatnonzero(window["High"].to_numpy() >= target_price)
            hit_target = len(hit_positions) > 0
            row[f"Hit Target {suffix}"] = "Yes" if hit_target else "No"
            row[f"Days to Target {suffix}"] = (
                int(hit_positions[0]) + 1 if hit_target else np.nan
            )
            row[f"Max High {suffix} (%)"] = (
                float(window["High"].max()) / entry_close - 1
            ) * 100

            if hit_target:
                drawdown_window = window["Low"].iloc[: int(hit_positions[0]) + 1]
            else:
                drawdown_window = window["Low"]
            row[f"Max Drawdown {suffix} (%)"] = (
                float(drawdown_window.min()) / entry_close - 1
            ) * 100

        rows.append(row)

    return pd.DataFrame(rows)


def build_results(
    data: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Build all signal group and EMA segment result tables."""

    results: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}

    for group in SIGNAL_GROUPS:
        cooldown_signal = apply_cooldown(data[group.signal_column], config.cooldown_candles)
        results[group.key] = {}

        for segment in EMA_SEGMENTS:
            eligible_signals = cooldown_signal & segment_mask(data, segment.key)
            signal_dates = data.index[eligible_signals]
            rows = [
                evaluate_signal_set(
                    data,
                    signal_dates,
                    holding_period,
                    config.target_return,
                )
                for holding_period in config.holding_periods
            ]
            results[group.key][segment.key] = {
                "summary": pd.DataFrame(rows),
                "details": build_signal_details(data, signal_dates, config),
            }

    return results


def format_metric(value: float | int, suffix: str = "") -> str:
    """Format metrics with graceful empty-state handling."""

    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}{suffix}"
    return f"{value:,.1f}{suffix}"


def parse_holding_periods(raw_value: str) -> tuple[int, ...]:
    """Parse comma-separated holding periods into sorted unique integers."""

    periods: list[int] = []
    for item in raw_value.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        if not cleaned.isdigit():
            raise ValueError("Holding periods must be positive whole numbers.")
        period = int(cleaned)
        if period <= 0:
            raise ValueError("Holding periods must be greater than zero.")
        if period > 252:
            raise ValueError("Holding periods must be 252 trading days or less.")
        periods.append(period)

    if not periods:
        raise ValueError("Enter at least one holding period, such as 3, 7, 14, 28.")

    return tuple(sorted(set(periods)))


def select_primary_period(holding_periods: tuple[int, ...]) -> int:
    """Choose the period used for headline metric cards and summary text."""

    if 14 in holding_periods:
        return 14
    return holding_periods[len(holding_periods) // 2]


def result_row_for_period(result_table: pd.DataFrame, holding_period: int) -> pd.Series:
    """Return one summary row for a numeric holding period."""

    label = f"{holding_period} Days"
    return result_table.loc[result_table["Holding Period"] == label].iloc[0]


def summarize_segment(
    group: SignalGroup,
    segment: EmaSegment,
    result_table: pd.DataFrame,
    config: BacktestConfig,
) -> str:
    """Create a compact statistical summary and recommendation."""

    primary_period = select_primary_period(config.holding_periods)
    longest_period = max(config.holding_periods)
    primary_row = result_row_for_period(result_table, primary_period)
    longest_row = result_row_for_period(result_table, longest_period)

    total_rounds = int(primary_row["Total Rounds"])
    if total_rounds == 0:
        return (
            f"{group.label} | {segment_label(segment, config)}: No complete historical "
            f"rounds were found for this segment in the selected {config.history_period} "
            "history window."
        )

    primary_win_rate = float(primary_row["Win Rate (%)"])
    primary_loss_rate = float(primary_row["Loss Rate (%)"])
    primary_drawdown = float(primary_row["Avg. Maximum Drawdown (%)"])
    primary_median_days = primary_row["Median Days to Target"]
    longest_win_rate = float(longest_row["Win Rate (%)"])

    recommendation = build_recommendation(
        group,
        segment,
        primary_win_rate,
        longest_win_rate,
        primary_drawdown,
        primary_period,
        longest_period,
    )

    return (
        f"{group.label} | {segment_label(segment, config)}: Total Rounds: {total_rounds}, "
        f"Win Rate ({primary_period} Days): {format_metric(primary_win_rate, '%')}, "
        f"Loss Rate: {format_metric(primary_loss_rate, '%')}, "
        f"Avg. Max Drawdown: {format_metric(primary_drawdown, '%')}, "
        f"Median Days: {format_metric(primary_median_days, ' days')}, "
        f"Win Rate ({longest_period} Days): {format_metric(longest_win_rate, '%')}. "
        f"{recommendation}"
    )


def build_recommendation(
    group: SignalGroup,
    segment: EmaSegment,
    primary_win_rate: float,
    longest_win_rate: float,
    primary_drawdown: float,
    primary_period: int,
    longest_period: int,
) -> str:
    """Generate a simple recommendation from the selected holding periods."""

    weak_drawdown = primary_drawdown <= -12
    strong_primary_edge = primary_win_rate >= 65
    strong_long_edge = longest_win_rate >= 70
    weak_edge = longest_win_rate < 50

    if segment.key == "above_ema" and strong_primary_edge:
        return (
            "Statistical recommendation: this stock is suitable for buying "
            "oversold RSI conditions during an uptrend, especially when the "
            f"{primary_period}-day target profile is the priority."
        )
    if segment.key == "above_ema" and strong_long_edge:
        return (
            "Statistical recommendation: the setup improves with a longer "
            f"{longest_period}-day window, so patience appears important after "
            "the RSI trigger."
        )
    if segment.key == "below_ema" and (weak_edge or weak_drawdown):
        return (
            "Statistical recommendation: avoid treating this as a simple dip-buy "
            "while price remains below the selected EMA; the historical profile "
            "resembles a falling-knife setup."
        )
    if group.key == "cross_up" and longest_win_rate >= 60:
        return (
            "Statistical recommendation: RSI recovery has historically offered "
            "a better confirmation signal when the holding window can extend up "
            f"to {longest_period} days."
        )
    if weak_edge:
        return (
            "Statistical recommendation: historical odds are not strong enough "
            "for this segment without additional confirmation."
        )

    return (
        "Statistical recommendation: use this as a watchlist signal and combine "
        "it with broader trend, liquidity, and risk controls."
    )


def styled_result_table(table: pd.DataFrame) -> pd.DataFrame:
    """Round displayed output without changing the underlying calculations."""

    display_table = table.copy()
    numeric_columns = [
        "Win Rate (%)",
        "Loss Rate (%)",
        "Avg. Maximum Drawdown (%)",
        "Median Days to Target",
    ]
    display_table[numeric_columns] = display_table[numeric_columns].round(1)
    return display_table


def styled_detail_table(table: pd.DataFrame) -> pd.DataFrame:
    """Round per-signal detail output for display."""

    if table.empty:
        return table

    display_table = table.copy()
    round_columns = [
        column
        for column in display_table.columns
        if column in {"Entry Close", "Target Price", "RSI", "EMA"}
        or column.endswith("(%)")
    ]
    display_table[round_columns] = display_table[round_columns].round(2)
    return display_table


def segment_label(segment: EmaSegment, config: BacktestConfig) -> str:
    """Render EMA segment labels with the selected EMA length."""

    if segment.key == "above_ema":
        return f"Above EMA {config.ema_length} (Buy the Dip)"
    if segment.key == "below_ema":
        return f"Below EMA {config.ema_length} (Falling Knife)"
    return segment.label


def render_segment(
    group: SignalGroup,
    segment: EmaSegment,
    table: pd.DataFrame,
    details: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """Render one EMA segment block."""

    st.subheader(segment_label(segment, config))

    primary_period = select_primary_period(config.holding_periods)
    longest_period = max(config.holding_periods)
    primary_row = result_row_for_period(table, primary_period)
    longest_row = result_row_for_period(table, longest_period)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Rounds", format_metric(primary_row["Total Rounds"]))
    metric_columns[1].metric(
        f"Win Rate {primary_period}D",
        format_metric(primary_row["Win Rate (%)"], "%"),
    )
    metric_columns[2].metric(
        f"Loss Rate {primary_period}D",
        format_metric(primary_row["Loss Rate (%)"], "%"),
    )
    metric_columns[3].metric(
        f"Avg. Max DD {primary_period}D",
        format_metric(primary_row["Avg. Maximum Drawdown (%)"], "%"),
    )
    metric_columns[4].metric(
        f"Win Rate {longest_period}D",
        format_metric(longest_row["Win Rate (%)"], "%"),
    )

    st.dataframe(styled_result_table(table), width="stretch", hide_index=True)

    with st.expander("Signal Details: days to target per round", expanded=False):
        if details.empty:
            st.caption("No completed signal rounds for this segment.")
        else:
            st.dataframe(
                styled_detail_table(details),
                width="stretch",
                hide_index=True,
            )

    st.info(summarize_segment(group, segment, table, config))


def render_app() -> None:
    """Main Streamlit entrypoint."""

    st.set_page_config(
        page_title="RSI/EMA Statistical Backtester",
        layout="wide",
    )

    st.title("RSI/EMA Statistical Backtester")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Stock Ticker", value="AAPL").strip().upper()
        rsi_length = st.number_input(
            "RSI length",
            min_value=2,
            max_value=100,
            value=RSI_LENGTH,
            step=1,
        )
        ema_length = st.number_input(
            "EMA length",
            min_value=2,
            max_value=500,
            value=EMA_LENGTH,
            step=1,
        )
        target_percent = st.number_input(
            "Target (%)",
            min_value=0.1,
            max_value=100.0,
            value=TARGET_RETURN * 100,
            step=0.5,
        )
        cooldown_candles = st.number_input(
            "Cooldown candles",
            min_value=0,
            max_value=250,
            value=COOLDOWN_CANDLES,
            step=1,
        )
        holding_periods_text = st.text_input(
            "Holding periods",
            value=", ".join(str(period) for period in HOLDING_PERIODS),
            help="Comma-separated trading days, for example: 3, 7, 14, 28",
        )
        history_period = st.selectbox(
            "History",
            options=HISTORY_OPTIONS,
            index=HISTORY_OPTIONS.index(YEARS_TO_FETCH),
        )

        try:
            holding_periods = parse_holding_periods(holding_periods_text)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        config = BacktestConfig(
            rsi_length=int(rsi_length),
            ema_length=int(ema_length),
            target_return=float(target_percent) / 100,
            cooldown_candles=int(cooldown_candles),
            history_period=history_period,
            holding_periods=holding_periods,
        )
        st.markdown(
            f"""
            **Rules**
            - RSI length: `{config.rsi_length}`
            - EMA length: `{config.ema_length}`
            - Target: `+{config.target_return:.1%}`
            - Cooldown: `{config.cooldown_candles}` candles
            - Holding periods: `{", ".join(str(period) for period in config.holding_periods)}`
            - History: `{config.history_period}`
            """
        )
        if PANDAS_TA_IMPORT_ERROR is not None:
            st.warning(
                "pandas-ta is installed but could not be imported in this Python "
                "environment, so the app is using equivalent pandas RSI/EMA "
                "calculations as a fallback."
            )

    st.caption(
        f"Tests RSI {RSI_TRIGGER_LEVEL} threshold events against a "
        f"+{config.target_return:.1%} forward high target over "
        f"{', '.join(str(period) for period in config.holding_periods)} "
        "trading-day windows."
    )

    if not ticker:
        st.warning("Enter a valid ticker to begin.")
        return

    try:
        with st.spinner(f"Fetching {ticker} historical data..."):
            price_data = download_price_history(ticker, config.history_period)

        if price_data.empty:
            st.error(f"No historical data was returned for ticker `{ticker}`.")
            return

        enriched_data = add_indicators(price_data, config)
        results = build_results(enriched_data, config)

    except Exception as exc:
        st.exception(exc)
        return

    start_date = enriched_data.index.min().date()
    end_date = enriched_data.index.max().date()
    st.success(
        f"Loaded {len(enriched_data):,} usable daily candles for `{ticker}` "
        f"from {start_date} to {end_date}."
    )

    with st.expander("Latest Indicator Snapshot", expanded=False):
        st.dataframe(
            enriched_data[["Open", "High", "Low", "Close", "RSI", "EMA"]]
            .tail(20)
            .round(2),
            width="stretch",
        )

    for group in SIGNAL_GROUPS:
        st.header(group.label)
        for segment in EMA_SEGMENTS:
            segment_results = results[group.key][segment.key]
            render_segment(
                group,
                segment,
                segment_results["summary"],
                segment_results["details"],
                config,
            )


if __name__ == "__main__":
    render_app()

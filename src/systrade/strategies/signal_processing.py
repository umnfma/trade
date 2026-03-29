"""
Signal processing modules: HMM regime detection + FFT cycle extraction.

These are composable building blocks used by the quantitative strategy.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

# Suppress hmmlearn convergence warnings during live fitting
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  HMM Regime Detector
# ═══════════════════════════════════════════════════════════════════════

class MarketRegime(Enum):
    MEAN_REVERTING = auto()
    TRENDING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class RegimeEstimate:
    regime: MarketRegime
    confidence: float       # 0–1, posterior probability of detected regime
    volatility: float       # current regime volatility estimate
    trend_strength: float   # positive = uptrend, negative = downtrend


class HMMRegimeDetector:
    """
    Fits a 3-state Gaussian HMM on rolling features to classify the
    current market micro-regime.

    Features per bar:
    1. 1-bar return
    2. Rolling volatility (std of last N returns)
    3. Volume ratio (current / rolling average)

    States are labeled post-hoc by their emission parameters:
    - Lowest variance → MEAN_REVERTING
    - Highest variance → VOLATILE
    - Middle → TRENDING
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback: int = 120,    # 2 hours of 1-min bars
        vol_window: int = 20,
        refit_interval: int = 30,  # refit every 30 bars
    ) -> None:
        self._n_states = n_states
        self._lookback = lookback
        self._vol_window = vol_window
        self._refit_interval = refit_interval

        self._returns: deque[float] = deque(maxlen=lookback)
        self._volumes: deque[float] = deque(maxlen=lookback)
        self._prices: deque[float] = deque(maxlen=lookback)
        self._bar_count = 0
        self._model: GaussianHMM | None = None
        self._state_map: dict[int, MarketRegime] = {}
        self._last_estimate = RegimeEstimate(
            MarketRegime.UNKNOWN, 0.0, 0.0, 0.0,
        )

    def update(self, price: float, volume: float) -> RegimeEstimate:
        """Feed a new bar and return the current regime estimate."""
        if not HMM_AVAILABLE:
            return RegimeEstimate(MarketRegime.UNKNOWN, 0.0, 0.0, 0.0)

        self._prices.append(price)
        self._volumes.append(volume)

        if len(self._prices) >= 2:
            ret = (price - self._prices[-2]) / self._prices[-2]
            self._returns.append(ret)

        self._bar_count += 1

        # Need enough data to fit
        if len(self._returns) < self._lookback:
            return self._last_estimate

        # Refit periodically
        if self._bar_count % self._refit_interval == 0:
            self._fit()

        if self._model is None:
            return self._last_estimate

        self._last_estimate = self._predict()
        return self._last_estimate

    def _fit(self) -> None:
        """Fit HMM on rolling feature matrix."""
        features = self._build_features()
        if features is None or len(features) < self._lookback:
            return

        try:
            model = GaussianHMM(
                n_components=self._n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            model.fit(features)
            self._model = model
            self._label_states()
        except Exception:
            pass  # fitting can fail on degenerate data

    def _predict(self) -> RegimeEstimate:
        """Predict current regime from latest features."""
        features = self._build_features()
        if features is None or self._model is None:
            return self._last_estimate

        try:
            posteriors = self._model.predict_proba(features)
            current_posterior = posteriors[-1]
            predicted_state = int(np.argmax(current_posterior))
            confidence = float(current_posterior[predicted_state])

            regime = self._state_map.get(predicted_state, MarketRegime.UNKNOWN)

            # Trend strength: mean of recent returns
            recent_rets = list(self._returns)[-20:]
            trend = sum(recent_rets) / len(recent_rets) if recent_rets else 0.0

            # Current vol
            vol = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0

            return RegimeEstimate(regime, confidence, vol, trend)
        except Exception:
            return self._last_estimate

    def _label_states(self) -> None:
        """Label HMM states by their emission variance."""
        if self._model is None:
            return
        # Use variance of the first feature (returns)
        variances = self._model.covars_[:, 0] if self._model.covars_.ndim == 2 \
            else [c[0, 0] for c in self._model.covars_]
        sorted_indices = np.argsort(variances)
        labels = [MarketRegime.MEAN_REVERTING, MarketRegime.TRENDING, MarketRegime.VOLATILE]
        self._state_map = {int(idx): labels[i] for i, idx in enumerate(sorted_indices)}

    def _build_features(self) -> np.ndarray | None:
        """Build feature matrix [returns, rolling_vol, volume_ratio]."""
        rets = list(self._returns)
        vols = list(self._volumes)

        if len(rets) < self._vol_window:
            return None

        n = len(rets)
        features = np.zeros((n, 3))
        features[:, 0] = rets

        # Rolling volatility
        for i in range(n):
            start = max(0, i - self._vol_window + 1)
            window = rets[start:i + 1]
            features[i, 1] = np.std(window) if len(window) > 1 else 0.0

        # Volume ratio
        avg_vol = np.mean(vols) if vols else 1.0
        for i in range(n):
            idx = len(vols) - n + i
            features[i, 2] = vols[idx] / avg_vol if avg_vol > 0 else 1.0

        return features


# ═══════════════════════════════════════════════════════════════════════
#  FFT Cycle Detector
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CycleEstimate:
    dominant_period: float     # bars per cycle (e.g., 45 = 45-minute cycle)
    phase: float               # 0 to 2π — where in the cycle we are
    amplitude: float           # strength of the cycle
    at_trough: bool            # near a buying point
    at_peak: bool              # near a selling point
    cycle_strength: float      # amplitude / noise ratio (signal quality)


class FFTCycleDetector:
    """
    Extracts the dominant intraday price cycle from VWAP deviations
    using FFT, then estimates current phase to predict turning points.

    Key insight: intraday price oscillations around VWAP often have
    semi-regular periodicity (institutional rebalancing, options hedging,
    market-maker inventory cycles).  FFT finds this hidden frequency.

    Usage: call update() with each bar's VWAP deviation.  The returned
    CycleEstimate tells you if price is near a trough (buy) or peak (sell).
    """

    def __init__(
        self,
        window: int = 120,        # bars for FFT (2 hours)
        min_period: int = 10,     # ignore cycles shorter than 10 min
        max_period: int = 90,     # ignore cycles longer than 90 min
        peak_threshold: float = 0.75,  # phase proximity to peak/trough (0-1)
    ) -> None:
        self._window = window
        self._min_period = min_period
        self._max_period = max_period
        self._peak_threshold = peak_threshold
        self._deviations: deque[float] = deque(maxlen=window)
        self._last_estimate = CycleEstimate(0, 0, 0, False, False, 0)

    def update(self, vwap_deviation: float) -> CycleEstimate:
        """Feed VWAP deviation for current bar, return cycle estimate."""
        self._deviations.append(vwap_deviation)

        if len(self._deviations) < self._window:
            return self._last_estimate

        self._last_estimate = self._analyze()
        return self._last_estimate

    def _analyze(self) -> CycleEstimate:
        """Run FFT and extract dominant cycle."""
        signal = np.array(self._deviations)

        # Detrend (remove linear trend)
        n = len(signal)
        x = np.arange(n)
        slope = (signal[-1] - signal[0]) / n if n > 1 else 0
        detrended = signal - (slope * x + signal[0])

        # Apply Hann window to reduce spectral leakage
        windowed = detrended * np.hanning(n)

        # FFT
        fft_result = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 bar

        # Magnitude spectrum (skip DC component)
        magnitudes = np.abs(fft_result[1:])
        phases = np.angle(fft_result[1:])
        freqs = freqs[1:]

        if len(magnitudes) == 0:
            return CycleEstimate(0, 0, 0, False, False, 0)

        # Filter to valid period range
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
        valid = (periods >= self._min_period) & (periods <= self._max_period)

        if not np.any(valid):
            return CycleEstimate(0, 0, 0, False, False, 0)

        valid_mags = magnitudes[valid]
        valid_phases = phases[valid]
        valid_periods = periods[valid]

        # Find dominant frequency
        peak_idx = np.argmax(valid_mags)
        dominant_period = float(valid_periods[peak_idx])
        amplitude = float(valid_mags[peak_idx])
        phase = float(valid_phases[peak_idx])

        # Normalize phase to 0–2π
        phase = phase % (2 * math.pi)

        # Noise floor: median of magnitudes outside dominant peak
        noise = float(np.median(valid_mags))
        cycle_strength = amplitude / noise if noise > 0 else 0.0

        # Determine if at peak or trough
        # Phase 0 = trough, π = peak (for cosine-like oscillation)
        # The current phase of the signal at the end of the window
        # determines where we are in the cycle
        at_trough = (phase > (2 * math.pi - math.pi * self._peak_threshold / 2)
                     or phase < (math.pi * self._peak_threshold / 2))
        at_peak = (math.pi * (1 - self._peak_threshold / 2) < phase
                   < math.pi * (1 + self._peak_threshold / 2))

        return CycleEstimate(
            dominant_period=dominant_period,
            phase=phase,
            amplitude=amplitude,
            at_trough=at_trough,
            at_peak=at_peak,
            cycle_strength=cycle_strength,
        )

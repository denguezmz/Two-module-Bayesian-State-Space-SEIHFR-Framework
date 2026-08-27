from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .utils import PROJECT_ROOT


# The Lancet/Elsevier artwork guidance recommends a standard, editable
# sans-serif typeface such as Arial or Helvetica.
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


FIGURE_DIR = PROJECT_ROOT / "figures"

PALETTE = {
    "blue": "#2C6EAA",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#7B61A8",
    "grey": "#6C757D",
    "light": "#E9EEF2",
    "dark": "#263238",
    "green": "#4C956C",
    "red": "#C44536",
}

SCENARIO_LABELS = {
    "Accelerated integrated response": "Accelerated response",
    "Current scale-up": "Continuation of current improvement",
    "Stalled response": "Maintenance of the current response level",
    "Operational disruption": "Response disruption",
}

SCENARIO_COLORS = {
    "Accelerated integrated response": PALETTE["green"],
    "Current scale-up": PALETTE["blue"],
    "Stalled response": PALETTE["gold"],
    "Operational disruption": PALETTE["red"],
}

STATIC_PARAM_LABELS = {
    "isolation_delay_base": "Baseline isolation delay",
    "followup_delay_reduction": "Follow-up delay reduction",
    "hospital_fatality_probability": "Hospital fatality probability",
    "hospital_outcome_days": "Hospital/isolation stay",
    "hospital_relative_infectiousness": "Hospital relative infectiousness",
    "funeral_relative_infectiousness": "Post-death relative infectiousness",
    "initial_exposed": "Initial exposed",
    "initial_infectious": "Initial infectious",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, alpha=0.22, linewidth=0.7)
    ax.tick_params(labelsize=8.5)


def _date_axis(ax: plt.Axes, interval: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, body: str, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 0.055, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color, transform=ax.transAxes)
    ax.text(x + width / 2, y + 0.095, body, ha="center", va="center", fontsize=8.9, color=PALETTE["dark"], linespacing=1.35, transform=ax.transAxes)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#455A64", width: float = 1.35, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, linewidth=width,
            color=color, transform=ax.transAxes, connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig1_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.035, 0.57), 0.17, 0.24, "DATA LAYER", "Official reports\nBackfill flags\nOperational indicators", PALETTE["dark"])
    _box(ax, (0.265, 0.57), 0.19, 0.24, "MODULE 1", "Event-time reconstruction\nLocal-level log Rt\nJoint FFBS paths", PALETTE["teal"])
    _box(ax, (0.525, 0.57), 0.20, 0.24, "MODULE 2", "SEIHFR cut posterior\nFull Rt path to beta(t)\nDeaths and stock update", PALETTE["orange"])
    _box(ax, (0.795, 0.57), 0.17, 0.24, "DECISION LAYER", "90-day scenarios\nDelay analysis\nSobol and PRCC", PALETTE["blue"])
    for a, b in [((0.205, 0.69), (0.265, 0.69)), ((0.455, 0.69), (0.525, 0.69)), ((0.725, 0.69), (0.795, 0.69))]:
        _arrow(ax, a, b)
    ax.text(0.50, 0.88, "One-way modular cut: Module 2 does not feed back into Module 1", ha="center", fontsize=9.5, color=PALETTE["grey"])
    _box(ax, (0.48, 0.17), 0.31, 0.18, "ROBUSTNESS CHECKS", "Backfill alternatives\nEffective population\nRandom seeds", PALETTE["purple"])
    _arrow(ax, (0.625, 0.57), (0.625, 0.35), color=PALETTE["purple"], width=1.15)
    ax.text(0.5, 0.955, "Modular Bayesian transmission-to-policy framework", ha="center", fontsize=16, weight="bold", color=PALETTE["dark"])
    ax.text(0.5, 0.07, "Core chain: event-time cases identify total Rt; SEIHFR explains compatible mechanisms and carries all forward policy simulations.", ha="center", fontsize=9.3, color=PALETTE["grey"])
    _save(fig, "Figure_01_Modular_Bayesian_framework")


def fig2_data(data: Any, config: dict) -> None:
    df = data.daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    anchors = data.anchors.copy()
    anchors["date"] = pd.to_datetime(anchors["date"])
    province = data.province.copy()
    sensitivity = data.case_backfill_sensitivity.copy()
    sensitivity["date"] = pd.to_datetime(sensitivity["date"])
    backfill_range = sensitivity.groupby("date")["reconstructed_incidence_cases"].agg(["min", "max"]).reset_index()

    fig = plt.figure(figsize=(13.8, 10.2))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.00, 1.08, 1.08],
        width_ratios=[1.00, 1.00],
        hspace=0.40,
        wspace=0.38,
    )

    ax = fig.add_subplot(gs[0, :])
    ax.plot(df["date"], df["cumulative_cases_reported"], color=PALETTE["blue"], linewidth=2.0, label="Reported cumulative cases")
    ax.plot(df["date"], df["cumulative_cases_event_time"], color=PALETTE["teal"], linestyle="--", linewidth=1.35, label="Backfill-adjusted cases")
    ax.scatter(anchors["date"], anchors["confirmed_cases"], s=18, color=PALETTE["blue"], alpha=0.55, label="Official case reports")
    ax.plot(df["date"], df["cumulative_deaths_reported"], color=PALETTE["red"], linewidth=1.8, label="Reported cumulative deaths")
    ax.plot(df["date"], df["cumulative_deaths_event_time"], color=PALETTE["orange"], linestyle="--", linewidth=1.35, label="Backfill-adjusted deaths")
    ax.scatter(anchors["date"], anchors["confirmed_deaths"], s=18, color=PALETTE["red"], alpha=0.55, label="Official death reports")
    ax.set_ylabel("Cumulative count")
    ax.set_title("A. Official cumulative reports and backfill adjustment", loc="left", weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_order = [
        "Reported cumulative cases",
        "Reported cumulative deaths",
        "Backfill-adjusted cases",
        "Backfill-adjusted deaths",
        "Official case reports",
        "Official death reports",
    ]
    ax.legend(
        [handle_by_label[label] for label in legend_order],
        legend_order,
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        loc="upper left",
        fontsize=7.7,
        columnspacing=0.9,
        handlelength=2.0,
    )
    _style(ax)
    ax.tick_params(axis="x", labelbottom=False)

    ax = fig.add_subplot(gs[1, :])
    ax.bar(df["date"], df["reported_incidence_cases"], width=0.80, color=PALETTE["grey"], alpha=0.35, label="Report-derived daily cases")
    ax.fill_between(backfill_range["date"], backfill_range["min"], backfill_range["max"], color=PALETTE["purple"], alpha=0.13, label="Backfill allocation range")
    ax.plot(df["date"], df["event_time_incidence_cases"], color=PALETTE["blue"], linewidth=1.7, label="Backfill-adjusted daily cases")
    positive = df["case_backfill_component"].to_numpy(float) > 0.01
    ax.fill_between(df["date"], 0, df["case_backfill_component"], where=positive, color=PALETTE["orange"], alpha=0.50, label="Notification-window backfill removed")
    ax.axvspan(pd.Timestamp(config["backfill"]["notification_start"]), pd.Timestamp(config["backfill"]["notification_end"]), color=PALETTE["gold"], alpha=0.16)
    ax.set_ylabel("Daily cases")
    ax.set_title("B. Explicit July backfill treatment and sensitivity envelope", loc="left", weight="bold")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    _style(ax)
    _date_axis(ax)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df["date"], 100 * df["contact_followup"], color=PALETTE["teal"], linewidth=2.0, label="Contact follow-up")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Follow-up (%)")
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["affected_health_zones"], color=PALETTE["purple"], linewidth=1.8, label="Affected health zones")
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("Affected health zones")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, ncol=1, frameon=False, loc="upper left", fontsize=7.8)
    ax.set_title("C. Operational coverage and geographic spread", loc="left", weight="bold")
    _style(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=8.5)
    _date_axis(ax)

    ax = fig.add_subplot(gs[2, 1])
    province_labels = {
        "Region A": "Ituri",
        "Region B": "North Kivu",
        "Region C": "Haut-Uele",
        "Region D": "Tshopo",
        "Region E": "South Kivu",
        "Region F": "Bas-Uele",
    }
    province["label"] = province["province"].map(province_labels).fillna(province["province"])
    province["case_share_pct"] = 100 * province["confirmed_cases"] / province["confirmed_cases"].sum()
    province["death_share_pct"] = 100 * province["confirmed_deaths"] / province["confirmed_deaths"].sum()
    province["bed_occupancy_pct"] = 100 * province["isolated_patients"] / province["documented_bed_capacity"]
    province = province.sort_values("confirmed_cases", ascending=False).reset_index(drop=True)
    y = np.arange(len(province))
    ax.scatter(province["case_share_pct"], y + 0.22, s=42, marker="o", color=PALETTE["blue"], label="Case share", zorder=3)
    ax.scatter(province["death_share_pct"], y, s=42, marker="s", color=PALETTE["red"], label="Death share", zorder=3)
    bed_mask = np.isfinite(province["bed_occupancy_pct"])
    ax.scatter(
        province.loc[bed_mask, "bed_occupancy_pct"], y[bed_mask] - 0.22,
        s=48, marker="D", facecolor="white", edgecolor=PALETTE["orange"],
        linewidth=1.4, label="Recorded bed occupancy", zorder=3,
    )
    ax.axvline(100, color=PALETTE["grey"], linestyle="--", linewidth=0.8, alpha=0.60)
    ax.set_xlim(0, 104)
    ax.set_yticks(y, province["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Share or occupancy (%)")
    ax.set_title("D. Provincial burden and recorded bed occupancy", loc="left", weight="bold")
    ax.legend(ncol=1, frameon=False, loc="lower right", fontsize=7.8)
    _style(ax, grid="x")

    fig.subplots_adjust(top=0.97, bottom=0.08, left=0.08, right=0.96)
    _save(fig, "Figure_1_Data_reconstruction")


def fig3_state(state_space: Any) -> None:
    fit = state_space.fit.copy()
    fit["date"] = pd.to_datetime(fit["date"])
    backfill = state_space.backfill_sensitivity_summary.copy()

    # Four equally weighted panels: transmission state, two fit checks, and
    # the latest-Rt sensitivity analysis across historical backfill schemes.
    fig = plt.figure(figsize=(14.0, 8.6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.00, 1.00],
        height_ratios=[1.00, 1.00],
        hspace=0.36,
        wspace=0.32,
    )
    ax_rt = fig.add_subplot(gs[0, 0])
    ax_rt.fill_between(
        fit["date"], fit["rt_smoothed_lo"], fit["rt_smoothed_hi"],
        color=PALETTE["teal"], alpha=0.17, label="95% credible interval",
    )
    ax_rt.plot(
        fit["date"], fit["rt_smoothed_median"],
        color=PALETTE["teal"], linewidth=2.1, label="Smoothed estimate",
    )
    ax_rt.plot(
        fit["date"], fit["rt_filtered_median"],
        color=PALETTE["grey"], linewidth=1.15, alpha=0.80,
        label="Filtered estimate",
    )
    ax_rt.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax_rt.set_ylabel(r"$R_t$")
    ax_rt.set_title("A. Estimated transmission state", loc="left", weight="bold")
    _style(ax_rt)
    _date_axis(ax_rt)

    ax_prob = ax_rt.twinx()
    ax_prob.plot(
        fit["date"], 100 * fit["prob_rt_gt_1"],
        color=PALETTE["blue"], linewidth=1.45, alpha=0.90,
        label=r"$P(R_t > 1)$",
    )
    ax_prob.set_ylim(-2, 102)
    ax_prob.set_yticks([0, 25, 50, 75, 100])
    ax_prob.set_ylabel(r"$P(R_t > 1)$, %", color=PALETTE["blue"])
    ax_prob.tick_params(axis="y", labelcolor=PALETTE["blue"], labelsize=8.5)
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_color(PALETTE["blue"])
    h1, l1 = ax_rt.get_legend_handles_labels()
    h2, l2 = ax_prob.get_legend_handles_labels()
    legend_a = ax_rt.legend(
        h1 + h2, l1 + l2, ncol=2, loc="upper right", fontsize=7.8,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
    )
    legend_a.set_zorder(10)

    ax = fig.add_subplot(gs[0, 1])
    daily = state_space.full_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["date"].isin(fit["date"])].copy()
    incidence = daily["event_time_incidence_cases"].astype(float).to_numpy()
    expected = fit["expected_incidence"].astype(float).to_numpy()
    expected_lo = fit["expected_incidence_lo"].astype(float).to_numpy()
    expected_hi = fit["expected_incidence_hi"].astype(float).to_numpy()
    ax.fill_between(fit["date"], expected_lo, expected_hi, color=PALETTE["teal"], alpha=0.18,
                    label="95% prediction interval")
    ax.plot(fit["date"], expected, color=PALETTE["teal"], linewidth=1.9,
            label="Model fit")
    ax.plot(fit["date"], incidence, color=PALETTE["blue"], linewidth=1.15,
            alpha=0.88, label="Backfill-adjusted observed cases")
    ax.set_ylabel("Daily cases")
    ax.set_title("B. Observed and fitted daily cases", loc="left", weight="bold")
    ax.legend(ncol=1, frameon=False, loc="upper left", fontsize=7.8)
    _style(ax)
    _date_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    cumulative_model = np.cumsum(np.nan_to_num(expected, nan=0.0))
    cumulative_model_lo = np.cumsum(np.nan_to_num(expected_lo, nan=0.0))
    cumulative_model_hi = np.cumsum(np.nan_to_num(expected_hi, nan=0.0))
    ax.fill_between(fit["date"], cumulative_model_lo, cumulative_model_hi, color=PALETTE["teal"], alpha=0.16,
                    label="95% prediction interval")
    ax.plot(fit["date"], cumulative_model, color=PALETTE["teal"], linewidth=1.9,
            label="Model fit")
    event_cum = daily["cumulative_cases_event_time"].astype(float).to_numpy()
    reported_cum = daily["cumulative_cases_reported"].astype(float).to_numpy()
    ax.plot(fit["date"], event_cum, color=PALETTE["blue"], linewidth=1.25,
            linestyle="--", label="Backfill-adjusted observed cases")
    ax.scatter(fit["date"], reported_cum, s=13, color=PALETTE["dark"], alpha=0.48,
               label="Official reports", zorder=3)
    ax.set_ylabel("Cumulative cases")
    ax.set_title("C. Observed and fitted cumulative cases", loc="left", weight="bold")
    ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=7.8)
    _style(ax)
    _date_axis(ax)

    ax = fig.add_subplot(gs[1, 1])
    backfill_labels = {
        "main_full_history_intensity": "Primary analysis",
        "sensitivity_a_28d_intensity": "28-day weighted",
        "sensitivity_b_42d_intensity": "42-day weighted",
        "sensitivity_c_56d_intensity": "56-day weighted",
        "sensitivity_d_full_history_uniform": "Full-history uniform",
        "sensitivity_e_notification_proportional": "Notification proportional",
    }
    backfill["label"] = backfill["scenario"].map(backfill_labels).fillna(backfill["scenario"])
    y = np.arange(len(backfill))
    for row_index, (_, row) in enumerate(backfill.iterrows()):
        is_primary = bool(row["is_primary"])
        color = PALETTE["teal"] if is_primary else PALETTE["grey"]
        ax.errorbar(
            row["rt_median"], row_index,
            xerr=np.array([[row["rt_median"] - row["rt_lower_95"]],
                           [row["rt_upper_95"] - row["rt_median"]]]),
            fmt="o", markersize=6.2 if is_primary else 5.2,
            color=color, ecolor=color, elinewidth=2.0 if is_primary else 1.5,
            capsize=3.0, markeredgecolor="white", markeredgewidth=0.7,
            zorder=3,
        )
    ax.axvline(1.0, color=PALETTE["red"], linestyle="--", linewidth=1.0,
               label=r"Control threshold ($R_t$ = 1)")
    ax.set_yticks(y, backfill["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.72, 1.48)
    ax.set_xlabel(r"$R_t$ on 19 Aug 2026")
    ax.set_title(r"D. Latest $R_t$ across backfill assumptions", loc="left", weight="bold")
    ax.legend(
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.015),
        borderaxespad=0.0, fontsize=8.0,
    )
    _style(ax, grid="x")

    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.98)
    _save(fig, "Figure_2_Transmission_state")


def _seihfr_diagram(ax: plt.Axes) -> None:
    ax.set_axis_off()
    positions = {
        "S": (0.035, 0.48), "E": (0.220, 0.48), "I": (0.405, 0.48),
        "H": (0.625, 0.67), "F": (0.625, 0.26), "R": (0.845, 0.67), "D": (0.845, 0.26),
    }
    width, height = 0.135, 0.150
    colors = {"S": "#BBD7F0", "E": "#F6E3A0", "I": "#F2B3A0", "H": "#A6DCD6", "F": "#C6B7DE", "R": "#B8D9C0", "D": "#C9D0D4"}
    labels = {
        "S": "S\nSusceptible", "E": "E\nExposed", "I": "I\nCommunity\ninfectious",
        "H": "H\nHospital /\nisolation", "F": "F\nPost-death", "R": "R\nRecovered", "D": "D\nDeaths",
    }

    def center(key: str) -> tuple[float, float]:
        x, y = positions[key]
        return x + width / 2, y + height / 2

    # Draw transitions first so all arrow shafts remain behind the state nodes.
    _arrow(ax, (positions["S"][0] + width, center("S")[1]), (positions["E"][0], center("E")[1]), width=1.25)
    _arrow(ax, (positions["E"][0] + width, center("E")[1]), (positions["I"][0], center("I")[1]), width=1.25)
    _arrow(ax, (positions["I"][0] + width, positions["I"][1] + 0.105), (positions["H"][0], positions["H"][1] + 0.045), color=PALETTE["blue"], width=1.25)
    _arrow(ax, (positions["I"][0] + width, positions["I"][1] + 0.040), (positions["F"][0], positions["F"][1] + 0.085), width=1.25)
    _arrow(ax, (positions["H"][0] + width, center("H")[1]), (positions["R"][0], center("R")[1]), width=1.25)
    _arrow(ax, (center("H")[0], positions["H"][1]), (center("F")[0], positions["F"][1] + height), width=1.10)
    _arrow(ax, (positions["F"][0] + width, center("F")[1]), (positions["D"][0], center("D")[1]), width=1.25)
    _arrow(
        ax,
        (center("I")[0], positions["I"][1] + height),
        (center("R")[0], positions["R"][1] + height),
        color=PALETTE["green"], width=1.05, rad=-0.22,
    )

    for key, (x, y) in positions.items():
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.009,rounding_size=0.025",
            facecolor=colors[key], edgecolor="#455A64", linewidth=1.15,
            transform=ax.transAxes, zorder=3,
        )
        ax.add_patch(patch)
        ax.text(*center(key), labels[key], ha="center", va="center", fontsize=7.0, weight="bold", linespacing=1.02, transform=ax.transAxes, zorder=4)

    legend_items = [
        (0.045, PALETTE["blue"], "Follow-up shortens I-to-H delay"),
        (0.405, PALETTE["teal"], "IPC reduces hospital transmission"),
        (0.790, PALETTE["green"], "Direct recovery"),
    ]
    for x, color, label in legend_items:
        ax.plot([x, x + 0.030], [0.105, 0.105], color=color, linewidth=2.7, transform=ax.transAxes, solid_capstyle="round")
        ax.text(x + 0.038, 0.105, label, ha="left", va="center", fontsize=6.8, color=color, transform=ax.transAxes)


def _plot_hospital_stock_calibration(ax: plt.Axes, fit: pd.DataFrame) -> None:
    ax.fill_between(
        fit["date"], fit["hospital_q2_5"], fit["hospital_q97_5"],
        color=PALETTE["purple"], alpha=0.16, label="95% posterior interval",
    )
    ax.plot(
        fit["date"], fit["hospital_median"],
        color=PALETTE["purple"], linewidth=1.9, label="Model estimate",
    )
    stock_mask = np.isfinite(fit["observed_hospital"])
    if stock_mask.any():
        ax.scatter(
            fit.loc[stock_mask, "date"], fit.loc[stock_mask, "observed_hospital"],
            s=38, facecolor="white", edgecolor=PALETTE["dark"], linewidth=1.1,
            label="Observed values", zorder=3,
        )
        ax.plot(
            fit.loc[stock_mask, "date"], fit.loc[stock_mask, "observed_hospital"],
            color=PALETTE["dark"], linewidth=0.9, alpha=0.45,
        )
    ax.set_ylabel("Hospital/isolation count")
    ax.set_title("D. Hospital/isolation reconstruction", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    _style(ax)
    _date_axis(ax)


def _save_fig4_subpanels(fit: pd.DataFrame, bridge: pd.DataFrame) -> None:
    fig_a, ax_a = plt.subplots(figsize=(9.0, 5.6))
    _seihfr_diagram(ax_a)
    ax_a.set_title("A. SEIHFR transmission and clinical-state structure", loc="left", weight="bold", fontsize=14)
    fig_a.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.96)
    _save(fig_a, "Figure_3A_SEIHFR_structure")

    fig_b, ax_b = plt.subplots(figsize=(9.2, 5.6))
    ax_b.fill_between(bridge["date"], bridge["state_space_rt_lower_95"], bridge["state_space_rt_upper_95"], color=PALETTE["teal"], alpha=0.14)
    ax_b.plot(bridge["date"], bridge["state_space_rt_median"], color=PALETTE["teal"], linewidth=2.0, label=r"Renewal-equation $R_t$")
    ax_b.fill_between(bridge["date"], bridge["seihfr_implied_rt_q2_5"], bridge["seihfr_implied_rt_q97_5"], color=PALETTE["orange"], alpha=0.13)
    ax_b.plot(bridge["date"], bridge["seihfr_implied_rt_median"], color=PALETTE["orange"], linewidth=1.8, label=r"SEIHFR-implied $R_t$")
    ax_b.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax_b.set_ylabel(r"$R_t$")
    ax_b.set_title(r"B. Full $R_t$-path bridge and mechanistic consistency", loc="left", weight="bold")
    ax_b.legend(frameon=False, ncol=2)
    _style(ax_b)
    _date_axis(ax_b)
    ax_bb = ax_b.twinx()
    ax_bb.plot(bridge["date"], bridge["beta_median"], color=PALETTE["purple"], linewidth=1.15, alpha=0.85)
    ax_bb.set_ylabel("beta(t)", color=PALETTE["purple"])
    ax_bb.tick_params(axis="y", labelcolor=PALETTE["purple"])
    ax_bb.spines["top"].set_visible(False)
    fig_b.subplots_adjust(top=0.90, bottom=0.17, left=0.09, right=0.89)
    _save(fig_b, "Figure_3B_Rt_bridge")

    fig_c, ax_c = plt.subplots(figsize=(9.0, 5.6))
    ax_c.fill_between(fit["date"], fit["cases_q2_5"], fit["cases_q97_5"], color=PALETTE["blue"], alpha=0.14)
    ax_c.plot(fit["date"], fit["cases_median"], color=PALETTE["blue"], linewidth=1.8, label="SEIHFR cases")
    ax_c.plot(fit["date"], fit["observed_cases"], color=PALETTE["dark"], linewidth=1.15, linestyle="--", label="Observed event-time cases")
    ax_c.fill_between(fit["date"], fit["deaths_q2_5"], fit["deaths_q97_5"], color=PALETTE["red"], alpha=0.10)
    ax_c.plot(fit["date"], fit["deaths_median"], color=PALETTE["red"], linewidth=1.6, label="SEIHFR deaths")
    ax_c.plot(fit["date"], fit["observed_deaths"], color=PALETTE["red"], linewidth=1.0, linestyle=":", label="Observed event-time deaths")
    ax_c.set_ylabel("Cumulative count")
    ax_c.set_title("C. Joint calibration to cases and deaths", loc="left", weight="bold")
    ax_c.legend(ncol=1, frameon=False, loc="upper left")
    _style(ax_c)
    _date_axis(ax_c)
    fig_c.subplots_adjust(top=0.90, bottom=0.17, left=0.11, right=0.97)
    _save(fig_c, "Figure_3C_joint_calibration")

    fig_d, ax_d = plt.subplots(figsize=(9.2, 5.4))
    _plot_hospital_stock_calibration(ax_d, fit)
    fig_d.subplots_adjust(top=0.88, bottom=0.18, left=0.12, right=0.97)
    _save(fig_d, "Figure_3D_hospital_stock_calibration")


def fig4_mechanism(calibration: Any) -> None:
    fit = calibration.fit_summary.copy()
    fit["date"] = pd.to_datetime(fit["date"])
    bridge = calibration.bridge_summary.copy()
    bridge["date"] = pd.to_datetime(bridge["date"])
    fig = plt.figure(figsize=(13.0, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1.0, 1.15], hspace=0.30, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    _seihfr_diagram(ax0)
    ax0.set_title("A. SEIHFR transmission and clinical-state structure", loc="left", weight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.fill_between(bridge["date"], bridge["state_space_rt_lower_95"], bridge["state_space_rt_upper_95"], color=PALETTE["teal"], alpha=0.14)
    ax1.plot(bridge["date"], bridge["state_space_rt_median"], color=PALETTE["teal"], linewidth=2.0, label=r"Renewal-equation $R_t$")
    ax1.fill_between(bridge["date"], bridge["seihfr_implied_rt_q2_5"], bridge["seihfr_implied_rt_q97_5"], color=PALETTE["orange"], alpha=0.13)
    ax1.plot(bridge["date"], bridge["seihfr_implied_rt_median"], color=PALETTE["orange"], linewidth=1.8, label=r"SEIHFR-implied $R_t$")
    ax1.axhline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax1.set_ylabel(r"$R_t$")
    ax1.set_title(r"B. Full $R_t$-path bridge and mechanistic consistency", loc="left", weight="bold")
    ax1.legend(frameon=False, ncol=2)
    _style(ax1)
    _date_axis(ax1)
    ax1b = ax1.twinx()
    ax1b.plot(bridge["date"], bridge["beta_median"], color=PALETTE["purple"], linewidth=1.15, alpha=0.85)
    ax1b.set_ylabel("beta(t)", color=PALETTE["purple"])
    ax1b.tick_params(axis="y", labelcolor=PALETTE["purple"])
    ax1b.spines["top"].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(fit["date"], fit["cases_q2_5"], fit["cases_q97_5"], color=PALETTE["blue"], alpha=0.14)
    ax2.plot(fit["date"], fit["cases_median"], color=PALETTE["blue"], linewidth=1.8, label="SEIHFR cases")
    ax2.plot(fit["date"], fit["observed_cases"], color=PALETTE["dark"], linewidth=1.15, linestyle="--", label="Observed event-time cases")
    ax2.fill_between(fit["date"], fit["deaths_q2_5"], fit["deaths_q97_5"], color=PALETTE["red"], alpha=0.10)
    ax2.plot(fit["date"], fit["deaths_median"], color=PALETTE["red"], linewidth=1.6, label="SEIHFR deaths")
    ax2.plot(fit["date"], fit["observed_deaths"], color=PALETTE["red"], linewidth=1.0, linestyle=":", label="Observed event-time deaths")
    ax2.set_ylabel("Cumulative count")
    ax2.set_title("C. Joint calibration to cases and deaths", loc="left", weight="bold")
    ax2.legend(ncol=1, frameon=False, loc="upper left")
    _style(ax2)
    _date_axis(ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    _plot_hospital_stock_calibration(ax3, fit)
    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.07, right=0.93, hspace=0.34, wspace=0.28)
    _save_fig4_subpanels(fit, bridge)
    _save(fig, "Figure_3_Mechanistic_reconstruction")


def fig_time_varying_route_contributions(mechanism: pd.DataFrame) -> None:
    """Save absolute and relative pathway contributions as two separate figures."""
    data = mechanism.copy()
    data["date"] = pd.to_datetime(data["date"])
    routes = [
        ("Community transmission", "community", PALETTE["teal"]),
        ("Transmission during hospitalisation or isolation", "hospital", PALETTE["orange"]),
        ("Post-death transmission", "postdeath", PALETTE["purple"]),
    ]

    def _plot(kind: str, title: str, ylabel: str, output_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        line_handles = []
        interval_handle = None
        upper_limit = 0.0
        for label, key, color in routes:
            if kind == "absolute":
                lower = data[f"R_{key}_q2_5"]
                median = data[f"R_{key}_median"]
                upper = data[f"R_{key}_q97_5"]
            else:
                lower = 100 * data[f"{key}_share_q2_5"]
                median = 100 * data[f"{key}_share_median"]
                upper = 100 * data[f"{key}_share_q97_5"]

            band = ax.fill_between(
                data["date"], lower, upper, color=color, alpha=0.10,
            )
            line, = ax.plot(
                data["date"], median, color=color, linewidth=2.1, label=label,
            )
            ax.scatter(
                data["date"].iloc[-1], median.iloc[-1], s=28,
                color=color, edgecolor="white", linewidth=0.6, zorder=3,
            )
            line_handles.append(line)
            interval_handle = band if interval_handle is None else interval_handle
            upper_limit = max(upper_limit, float(upper.max()))

        ax.set_ylim(0.0, 1.05 * upper_limit)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.legend(
            line_handles + [interval_handle],
            [label for label, _, _ in routes] + ["95% posterior intervals"],
            ncol=2, loc="upper right", frameon=True, facecolor="white",
            edgecolor="none", framealpha=0.94, fontsize=8.0,
        )
        _style(ax)
        _date_axis(ax)
        fig.subplots_adjust(top=0.94, bottom=0.17, left=0.10, right=0.98)
        _save(fig, output_name)

    _plot(
        "absolute", r"A. Contribution to $R_t$", r"Contribution to $R_t$",
        "Figure_Time_varying_pathway_contributions_absolute",
    )
    _plot(
        "relative", "B. Relative contribution", "Contribution share (%)",
        "Figure_Time_varying_pathway_contributions_relative",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    legend_lines = []
    interval_handle = None
    for axis_index, (ax, kind, title, ylabel) in enumerate([
        (axes[0], "absolute", r"A. Contribution to $R_t$", r"Contribution to $R_t$"),
        (axes[1], "relative", "B. Relative contribution", "Contribution share (%)"),
    ]):
        upper_limit = 0.0
        for label, key, color in routes:
            if kind == "absolute":
                lower = data[f"R_{key}_q2_5"]
                median = data[f"R_{key}_median"]
                upper = data[f"R_{key}_q97_5"]
            else:
                lower = 100 * data[f"{key}_share_q2_5"]
                median = 100 * data[f"{key}_share_median"]
                upper = 100 * data[f"{key}_share_q97_5"]
            band = ax.fill_between(
                data["date"], lower, upper, color=color, alpha=0.08,
            )
            line, = ax.plot(
                data["date"], median, color=color, linewidth=2.0, label=label,
            )
            ax.scatter(
                data["date"].iloc[-1], median.iloc[-1], s=26,
                color=color, edgecolor="white", linewidth=0.6, zorder=3,
            )
            if axis_index == 0:
                legend_lines.append(line)
                interval_handle = band if interval_handle is None else interval_handle
            upper_limit = max(upper_limit, float(upper.max()))
        ax.set_ylim(0.0, 1.05 * upper_limit)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        _style(ax)
        _date_axis(ax)

    axes[0].legend(
        legend_lines + [interval_handle],
        [label for label, _, _ in routes] + ["95% posterior intervals"],
        ncol=2,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        fontsize=7.8,
        handlelength=2.3,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.07, right=0.98, wspace=0.20)
    _save(fig, "Figure_Time_varying_pathway_contributions")


def fig_route_contributions_identifiability() -> None:
    """Combine pathway contributions, posterior learning, and initial-state compensation."""
    mechanism = pd.read_csv(PROJECT_ROOT / "results" / "seihfr_mechanism_decomposition_daily.csv")
    learning = pd.read_csv(PROJECT_ROOT / "results" / "seihfr_prior_posterior_learning.csv")
    corr = pd.read_csv(PROJECT_ROOT / "results" / "seihfr_parameter_correlation.csv")
    particles = pd.read_csv(PROJECT_ROOT / "results" / "seihfr_cut_posterior_particles.csv")

    mechanism["date"] = pd.to_datetime(mechanism["date"])
    routes = [
        ("Community transmission", "community", PALETTE["teal"]),
        ("Transmission during hospitalisation or isolation", "hospital", PALETTE["orange"]),
        ("Post-death transmission", "postdeath", PALETTE["purple"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    legend_handles = []
    interval_handle = None
    upper_limit = 0.0
    for label, key, color in routes:
        lower = mechanism[f"R_{key}_q2_5"]
        median = mechanism[f"R_{key}_median"]
        upper = mechanism[f"R_{key}_q97_5"]
        band = ax_a.fill_between(mechanism["date"], lower, upper, color=color, alpha=0.10)
        line, = ax_a.plot(mechanism["date"], median, color=color, linewidth=1.9, label=label)
        ax_a.scatter(mechanism["date"].iloc[-1], median.iloc[-1], s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        legend_handles.append(line)
        interval_handle = band if interval_handle is None else interval_handle
        upper_limit = max(upper_limit, float(upper.max()))
    ax_a.axhline(1.0, color=PALETTE["grey"], linestyle="--", linewidth=0.8, alpha=0.75)
    ax_a.set_ylim(0.0, 1.05 * upper_limit)
    ax_a.set_ylabel(r"Contribution to $R_t$")
    ax_a.set_title("A. Time-varying pathway contributions", loc="left", weight="bold")
    ax_a.legend(
        legend_handles + [interval_handle],
        [label for label, _, _ in routes] + ["95% posterior interval"],
        ncol=1,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        fontsize=7.4,
        handlelength=2.1,
        columnspacing=0.8,
        labelspacing=0.22,
    )
    _style(ax_a)
    _date_axis(ax_a)

    for label, key, color in routes:
        lower = mechanism[f"{key}_share_q2_5"]
        median = mechanism[f"{key}_share_median"]
        upper = mechanism[f"{key}_share_q97_5"]
        ax_b.fill_between(mechanism["date"], lower, upper, color=color, alpha=0.10)
        ax_b.plot(mechanism["date"], median, color=color, linewidth=1.9)
        ax_b.scatter(mechanism["date"].iloc[-1], median.iloc[-1], s=24, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_b.set_ylim(0, 1)
    ax_b.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_b.set_ylabel("Contribution share")
    ax_b.set_title("B. Relative contribution shares", loc="left", weight="bold")
    _style(ax_b)
    _date_axis(ax_b)

    learning = learning[~learning["parameter"].str.startswith("beta_spline")].copy()
    learning["label"] = learning["parameter"].map(STATIC_PARAM_LABELS).fillna(learning["parameter"])
    limited_params = {
        "followup_delay_reduction",
        "isolation_delay_base",
        "hospital_relative_infectiousness",
        "funeral_relative_infectiousness",
    }
    learning = learning.sort_values("interval_contraction_ratio", ascending=True)
    bar_colors = [
        PALETTE["orange"] if param in limited_params else PALETTE["blue"]
        for param in learning["parameter"]
    ]
    ax_c.barh(learning["label"], learning["interval_contraction_ratio"], color=bar_colors, alpha=0.86)
    ax_c.axvline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax_c.set_xlabel("Prior 95% width / posterior 95% width")
    ax_c.set_title("C. Posterior learning in mechanism parameters", loc="left", weight="bold")
    ax_c.text(
        1.03,
        0.97,
        "limited\nshrinkage",
        transform=ax_c.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.8,
        color=PALETTE["grey"],
    )
    ax_c.invert_yaxis()
    _style(ax_c, grid="x")

    if "posterior_weight" in particles:
        weights = particles["posterior_weight"].to_numpy()
        weights = weights / weights.sum()
    else:
        weights = None
    plot_particles = particles.sample(
        n=min(len(particles), 1800),
        weights=weights,
        random_state=20260819,
    )
    ax_d.scatter(
        plot_particles["initial_exposed"],
        plot_particles["initial_infectious"],
        s=13,
        color=PALETTE["blue"],
        alpha=0.18,
        edgecolor="none",
    )
    x = particles["initial_exposed"].to_numpy()
    y = particles["initial_infectious"].to_numpy()
    if weights is not None:
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    else:
        slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
    ax_d.plot(x_line, slope * x_line + intercept, color=PALETTE["red"], linewidth=1.6)
    mask = (
        (corr["parameter_a"].eq("initial_exposed") & corr["parameter_b"].eq("initial_infectious"))
        | (corr["parameter_a"].eq("initial_infectious") & corr["parameter_b"].eq("initial_exposed"))
    )
    r_value = float(corr.loc[mask, "weighted_correlation"].iloc[0]) if mask.any() else float(np.corrcoef(x, y)[0, 1])
    ax_d.text(
        0.05,
        0.92,
        f"Posterior r = {r_value:.2f}",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=PALETTE["dark"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    ax_d.set_xlabel("Initial exposed")
    ax_d.set_ylabel("Initial infectious")
    ax_d.set_title("D. Compensation between early infection states", loc="left", weight="bold")
    _style(ax_d)

    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, hspace=0.34, wspace=0.28)
    _save(fig, "Figure_4_Pathway_contributions_and_identifiability")


def fig4b_mechanism_fit(calibration: Any) -> None:
    fit = calibration.fit_summary.copy()
    fit["date"] = pd.to_datetime(fit["date"])
    metrics = calibration.fit_metrics.set_index("target")

    def _metric_text(target: str) -> str:
        if target not in metrics.index:
            return ""
        row = metrics.loc[target]
        return f"WAPE={row['WAPE']:.1%}  Coverage95={row['coverage_95']:.1%}"

    fig, axes = plt.subplots(3, 1, figsize=(11.4, 10.6), sharex=True)

    ax = axes[0]
    ax.fill_between(fit["date"], fit["cases_q2_5"], fit["cases_q97_5"], color=PALETTE["blue"], alpha=0.18)
    ax.plot(fit["date"], fit["cases_median"], color=PALETTE["blue"], linewidth=1.9, label="SEIHFR median")
    ax.plot(fit["date"], fit["observed_cases"], color=PALETTE["dark"], linewidth=1.1, linestyle="--", alpha=0.85, label="Observed cumulative cases")
    ax.set_ylabel("Cumulative cases")
    ax.set_title("A. Case cumulative fit", loc="left", weight="bold")
    ax.text(0.02, 0.95, _metric_text("cases"), transform=ax.transAxes, fontsize=8.6, color=PALETTE["grey"], va="top")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.0)
    _style(ax)
    _date_axis(ax)

    ax = axes[1]
    ax.fill_between(fit["date"], fit["deaths_q2_5"], fit["deaths_q97_5"], color=PALETTE["red"], alpha=0.15, label="95% posterior interval")
    ax.plot(fit["date"], fit["deaths_median"], color=PALETTE["red"], linewidth=1.8, label="SEIHFR median")
    ax.plot(fit["date"], fit["observed_deaths"], color=PALETTE["dark"], linewidth=1.0, linestyle=":", alpha=0.82, label="Observed cumulative deaths")
    ax.set_ylabel("Cumulative deaths")
    ax.set_title("B. Death cumulative fit", loc="left", weight="bold")
    ax.text(0.02, 0.95, _metric_text("deaths"), transform=ax.transAxes, fontsize=8.6, color=PALETTE["grey"], va="top")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.0)
    _style(ax)
    _date_axis(ax)

    ax = axes[2]
    ax.fill_between(fit["date"], fit["hospital_q2_5"], fit["hospital_q97_5"], color=PALETTE["purple"], alpha=0.16, label="95% posterior interval")
    ax.plot(fit["date"], fit["hospital_median"], color=PALETTE["purple"], linewidth=1.8, label="SEIHFR median stock")
    stock_mask = np.isfinite(fit["observed_hospital"])
    if stock_mask.any():
        ax.scatter(
            fit.loc[stock_mask, "date"],
            fit.loc[stock_mask, "observed_hospital"],
            s=38,
            facecolor="white",
            edgecolor=PALETTE["dark"],
            linewidth=1.1,
            label="Observed stock values",
            zorder=3,
        )
        ax.plot(
            fit.loc[stock_mask, "date"],
            fit.loc[stock_mask, "observed_hospital"],
            color=PALETTE["dark"],
            linewidth=0.9,
            alpha=0.45,
        )
    ax.set_ylabel("Hospital/isolation stock")
    ax.set_title("C. Hospital/isolation stock fit", loc="left", weight="bold")
    ax.text(0.02, 0.95, _metric_text("hospital"), transform=ax.transAxes, fontsize=8.6, color=PALETTE["grey"], va="top")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.0)
    _style(ax)
    _date_axis(ax)

    fig.suptitle("Mechanistic posterior predictive fit diagnostics", fontsize=15, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.985])
    _save(fig, "Figure_04B_Mechanistic_fit_diagnostics")


def fig5_policy(scenarios: Any) -> None:
    daily = scenarios.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    delay = scenarios.delay.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    scenario_order = [
        "Accelerated integrated response",
        "Current scale-up",
        "Stalled response",
        "Operational disruption",
    ]
    for scenario, group in daily.groupby("scenario"):
        g = group[group["forecast_day"] <= 90]
        color = SCENARIO_COLORS.get(scenario, PALETTE["grey"])
        axes[0, 0].fill_between(g["date"], g["cases_q25"], g["cases_q75"], color=color, alpha=0.12)
        axes[0, 0].plot(g["date"], g["cases_median"], color=color, linewidth=1.9, label=SCENARIO_LABELS.get(scenario, scenario))
        axes[0, 1].plot(g["date"], g["incidence_median"], color=color, linewidth=1.6, label=SCENARIO_LABELS.get(scenario, scenario))
        axes[1, 0].plot(g["date"], g["hospital_median"], color=color, linewidth=1.7, label=SCENARIO_LABELS.get(scenario, scenario))
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Cumulative cases, log scale")
    axes[0, 0].set_title("A. Ninety-day cumulative cases", loc="left", weight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_by_label = dict(zip(labels, handles))
    ordered_labels = [SCENARIO_LABELS[s] for s in scenario_order]
    axes[0, 0].legend(
        [legend_by_label[label] for label in ordered_labels if label in legend_by_label],
        [label for label in ordered_labels if label in legend_by_label],
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=7.6,
        handlelength=1.8,
        columnspacing=1.15,
        labelspacing=0.18,
        borderaxespad=0.25,
    )
    axes[0, 1].set_yscale("symlog", linthresh=20)
    axes[0, 1].set_ylabel("Daily cases, symlog scale")
    axes[0, 1].set_title("B. Case flow separates across operational paths", loc="left", weight="bold")
    axes[1, 0].axhline(837, color=PALETTE["dark"], linestyle="--", linewidth=0.9, label="Current stock 837")
    axes[1, 0].set_yscale("symlog", linthresh=500)
    axes[1, 0].set_ylabel("Hospital/isolation stock")
    axes[1, 0].set_title("C. Hospital and isolation pressure", loc="left", weight="bold")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend_by_label = dict(zip(labels, handles))
    ordered_labels = [
        "Accelerated response",
        "Continuation of current improvement",
        "Maintenance of the current response level",
        "Response disruption",
        "Current stock 837",
    ]
    axes[1, 0].legend(
        [legend_by_label[label] for label in ordered_labels if label in legend_by_label],
        [label for label in ordered_labels if label in legend_by_label],
        frameon=False,
        ncol=2,
        loc="lower left",
        fontsize=8.0,
        handlelength=1.9,
        columnspacing=1.15,
        labelspacing=0.25,
    )
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("Date")
        _style(ax)
        _date_axis(ax)

    ax = axes[1, 1]
    ax.plot(delay["delay_days"], delay["cases90_median"], marker="o", color=PALETTE["orange"], linewidth=2.0)
    ax.set_ylim(delay["cases90_median"].min() * 0.965, delay["cases90_median"].max() * 1.075)
    for _, row in delay.iterrows():
        label = "baseline" if row["delay_days"] == 0 else f"+{row['additional_cases90_median']:.0f}"
        ax.text(row["delay_days"], row["cases90_median"] * 1.015, label, ha="center", fontsize=8.1)
    ax.set_xlabel("Delay in accelerated response, days")
    ax.set_ylabel("Day-90 cumulative cases")
    ax.set_title("D. Delayed acceleration increases cumulative cases", loc="left", weight="bold")
    _style(ax)
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, "Figure_5_Policy_scenarios")


def fig6_sensitivity(sensitivity: Any) -> None:
    idx = sensitivity.indices.sort_values("total_order_sobol_cases90", ascending=True).copy()
    y = np.arange(len(idx))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), gridspec_kw={"width_ratios": [1.0, 1.22]})
    ax = axes[0]
    ax.barh(y, idx["total_order_sobol_cases90"], color=PALETTE["orange"], alpha=0.84, label="Total effect")
    ax.barh(y, idx["first_order_sobol_cases90"], color=PALETTE["blue"], alpha=0.82, label="First order")
    ax.set_yticks(y, idx["parameter"], fontsize=8)
    ax.set_xlabel("Sobol index for day-90 cases")
    ax.set_title("A. Variance contribution to cumulative cases", loc="left", weight="bold")
    ax.legend(frameon=False)
    _style(ax, grid="x")

    ax = axes[1]
    h = 0.22
    ax.barh(y + h, idx["PRCC_cases90"], height=h, color=PALETTE["blue"], label="Day-90 cases")
    ax.barh(y, idx["PRCC_deaths90"], height=h, color=PALETTE["red"], label="Day-90 deaths")
    ax.barh(y - h, idx["PRCC_peak_hospital90"], height=h, color=PALETTE["purple"], label="Peak stock")
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_yticks(y, idx["parameter"], fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("PRCC")
    ax.set_title("B. Direction and strength across outcomes", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    _style(ax, grid="x")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "Figure_6_SEIHFR_policy_sensitivity")


def fig_s5_effective_population(neff: pd.DataFrame | None = None) -> None:
    if neff is None:
        path = PROJECT_ROOT / "results" / "seihfr_effective_population_sensitivity.csv"
        if not path.exists():
            return
        neff = pd.read_csv(path)
    if neff.empty:
        return
    df = neff.copy()
    df["n_eff_millions"] = df["effective_population"].astype(float) / 1_000_000.0
    panels = [
        ("cases90_median", "Day-90 cumulative cases"),
        ("deaths90_median", "Day-90 cumulative deaths"),
        ("peak_hospital90_median", "Peak hospital/isolation stock"),
        ("rt90_median", "Day-90 Rt"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    for ax, (column, title) in zip(axes.flat, panels):
        for scenario, group in df.groupby("scenario"):
            g = group.sort_values("n_eff_millions")
            color = SCENARIO_COLORS.get(scenario, PALETTE["grey"])
            ax.plot(
                g["n_eff_millions"],
                g[column],
                marker="o",
                linewidth=1.9,
                color=color,
                label=SCENARIO_LABELS.get(scenario, scenario),
            )
        ax.axvline(5.0, color=PALETTE["dark"], linestyle="--", linewidth=0.85, alpha=0.65)
        ax.set_xlabel("Effective connected population, millions")
        ax.set_title(title, loc="left", weight="bold")
        _style(ax)
    axes[0, 0].set_ylabel("Count")
    axes[0, 1].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    axes[1, 1].set_ylabel("Rt")
    axes[1, 1].axhline(1.0, color=PALETTE["red"], linestyle=":", linewidth=0.9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Conditional sensitivity to effective connected population", fontsize=15, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.965])
    _save(fig, "Figure_S5_Effective_population_sensitivity")


def supplements(calibration: Any, neff: pd.DataFrame | None = None) -> None:
    learning = calibration.prior_posterior_learning.copy()
    learning = learning[~learning["parameter"].str.startswith("beta_spline")].copy()
    learning["label"] = learning["parameter"].map(STATIC_PARAM_LABELS).fillna(learning["parameter"])
    learning = learning.sort_values("interval_contraction_ratio")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.barh(learning["label"], learning["interval_contraction_ratio"], color=PALETTE["blue"], alpha=0.82)
    ax.axvline(1.0, color=PALETTE["grey"], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Prior 95% width / posterior 95% width")
    ax.set_title("Prior-to-posterior learning for non-beta SEIHFR parameters", loc="left", weight="bold")
    _style(ax, grid="x")
    fig.tight_layout()
    _save(fig, "Figure_S1_Prior_posterior_learning")

    corr = calibration.parameter_correlation.head(15).iloc[::-1].copy()
    corr["pair"] = corr["parameter_a"].map(STATIC_PARAM_LABELS).fillna(corr["parameter_a"]) + " vs " + corr["parameter_b"].map(STATIC_PARAM_LABELS).fillna(corr["parameter_b"])
    colors = [PALETTE["red"] if z < 0 else PALETTE["blue"] for z in corr["weighted_correlation"]]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.barh(corr["pair"], corr["weighted_correlation"], color=colors, alpha=0.86)
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Posterior weighted correlation")
    ax.set_title("Largest posterior correlations among static mechanism parameters", loc="left", weight="bold")
    _style(ax, grid="x")
    fig.tight_layout()
    _save(fig, "Figure_S3_Parameter_correlation")

    obs = calibration.calibration_observations.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    stock = obs[obs["target"] == "hospital"].copy()
    if not stock.empty:
        fit = calibration.fit_summary.copy()
        fit["date"] = pd.to_datetime(fit["date"])
        fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.25, 0.75]})
        ax = axes[0]
        ax.fill_between(fit["date"], fit["hospital_q2_5"], fit["hospital_q97_5"], color=PALETTE["purple"], alpha=0.16, label="SEIHFR 95% interval")
        ax.plot(fit["date"], fit["hospital_median"], color=PALETTE["purple"], linewidth=2.0, label="SEIHFR median")
        ax.scatter(stock["date"], stock["observed"], s=44, facecolor="white", edgecolor=PALETTE["dark"], linewidth=1.2, label="Observed stock values", zorder=3)
        ax.plot(stock["date"], stock["observed"], color=PALETTE["dark"], linewidth=0.9, alpha=0.45)
        ax.set_ylabel("Hospital/isolation stock")
        ax.set_title("A. Hospital/isolation stock posterior predictive check", loc="left", weight="bold")
        ax.legend(frameon=False, ncol=3)
        _style(ax)
        ax = axes[1]
        ax.axhline(0, color=PALETTE["dark"], linewidth=0.8)
        colors = [PALETTE["green"] if inside else PALETTE["red"] for inside in stock["inside_95"]]
        ax.bar(stock["date"], 100 * stock["relative_error"], width=2.5, color=colors, alpha=0.78)
        ax.set_ylabel("Median error, %")
        ax.set_title("B. Stock residuals by report date", loc="left", weight="bold")
        _style(ax)
        _date_axis(ax)
        fig.suptitle("Focused calibration diagnostic for sparse hospital/isolation stock reports", fontsize=14.5, weight="bold", y=0.995)
        fig.tight_layout()
        _save(fig, "Figure_S4_Hospital_stock_calibration")
    fig_s5_effective_population(neff)


def generate_all_figures(data: Any, state_space: Any, calibration: Any, scenarios: Any, sensitivity: Any, config: dict, neff: pd.DataFrame | None = None) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.3,
        "figure.dpi": 135,
        "axes.unicode_minus": False,
    })
    fig2_data(data, config)
    fig3_state(state_space)
    fig4_mechanism(calibration)
    fig4b_mechanism_fit(calibration)
    fig5_policy(scenarios)
    fig6_sensitivity(sensitivity)
    supplements(calibration, neff)

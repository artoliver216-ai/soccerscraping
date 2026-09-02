"""Scrape completed-match xG data for the last two full Premier League seasons
(2024/25 and 2025/26) from Understat, using a headless browser since Understat
renders its match calendar via JavaScript.

Understat's calendar only shows one week at a time, so for each season we start
at the most recent week and click "prev week" repeatedly, collecting matches,
until the button disables (start of the season). A 3-second delay is inserted
before each "prev week" click to stay under rate limits.
"""

import sys
import time

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

SEASONS = ["2024", "2025"]  # understat labels seasons by their starting year:
# "2024" -> 2024/25 season, "2025" -> 2025/26 season
BASE_URL = "https://understat.com/league/EPL/{season}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
OUTPUT_CSV = "fbref_xg.csv"
REQUEST_DELAY_SECONDS = 3
MAX_WEEKS_PER_SEASON = 60  # safety cap; a season is ~40 calendar weeks
STALL_LIMIT = 3  # stop early if several consecutive clicks add no new matches


def parse_matches(html, season):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for date_container in soup.select(".calendar-date-container"):
        date_el = date_container.select_one(".calendar-date")
        date = date_el.get_text(strip=True) if date_el else None

        for game in date_container.select(".calendar-game"):
            match_info = game.select_one('a.match-info[data-isresult="true"]')
            if match_info is None:
                continue  # not yet played

            match_id = match_info.get("href", "").split("/")[-1]
            home_name = game.select_one(".block-home .team-title a").get_text(strip=True)
            away_name = game.select_one(".block-away .team-title a").get_text(strip=True)

            goals = match_info.select(".teams-goals span")
            xg = match_info.select(".teams-xG span")
            home_goals, away_goals = goals[0].get_text(strip=True), goals[1].get_text(strip=True)
            home_xg, away_xg = xg[0].get_text(strip=True), xg[1].get_text(strip=True)

            rows.append(
                {
                    "MatchID": match_id,
                    "Season": f"{season}/{str(int(season) + 1)[-2:]}",
                    "Date": date,
                    "Home": home_name,
                    "Home_xG": home_xg,
                    "Score": f"{home_goals}-{away_goals}",
                    "Away_xG": away_xg,
                    "Away": away_name,
                }
            )
    return rows


def scrape_season(page, season):
    url = BASE_URL.format(season=season)
    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_selector(".calendar-container", timeout=15000)

    matches_by_id = {}
    stall_count = 0

    for _ in range(MAX_WEEKS_PER_SEASON):
        html = page.inner_html(".calendar-container")
        before = len(matches_by_id)
        for row in parse_matches(html, season):
            matches_by_id[row["MatchID"]] = row
        stall_count = stall_count + 1 if len(matches_by_id) == before else 0

        prev_button = page.query_selector(".calendar-prev")
        if prev_button is None or prev_button.get_attribute("disabled") is not None:
            break
        if stall_count >= STALL_LIMIT:
            break

        time.sleep(REQUEST_DELAY_SECONDS)  # rate limit before the next "request"
        prev_button.click()
        page.wait_for_timeout(800)  # let the week's data render

    return list(matches_by_id.values())


def main():
    all_rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for i, season in enumerate(SEASONS):
            if i > 0:
                time.sleep(REQUEST_DELAY_SECONDS)
            page = browser.new_page(user_agent=USER_AGENT)
            rows = scrape_season(page, season)
            print(f"Season {season}: {len(rows)} completed matches")
            all_rows.extend(rows)
            page.close()
        browser.close()

    if not all_rows:
        sys.exit("Could not find any completed matches with xG data.")

    df = pd.DataFrame(all_rows).drop(columns=["MatchID"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

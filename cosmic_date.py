from typing import List, Tuple, Sequence, NamedTuple
from constants import SECS_IN_DAY, SECS_IN_YEAR, GREAT_YEAR, YEAR, DAY, MONTH, PRECESSION_ANGLE_SEC, PRECESSION_ANGLE_HOUR, PRECESSION_ANGLE_MINUTE, PRECESSION_ANGLE_DAY, PRECESSION_ANGLE_YEAR, PRECESSION_ANGLE_MONTH

class CosmicDate(NamedTuple):
  year:int
  month:int
  day:int
  hour:int
  minute:int
  second:int


def valid_year(year): return 0 <= (year-1) < GREAT_YEAR
def valid_month(month): return 0 <= (month-1) < YEAR
def valid_day(day): return 0 <= (day-1) < MONTH
def valid_hour(hour): return 0 <= (hour-1) < DAY
def valid_minute(minute): return 0 <= (minute-1) < 60
def valid_sec(sec): return 0 <= (sec-1) < 60

def valid_date(date: CosmicDate):
    year, month, day, hour, minute, sec = date
    return all([valid_year(year), valid_month(month), valid_day(day),
                valid_hour(hour), valid_minute(minute), valid_sec(sec)])

def cosmic_date_in_seconds(date: CosmicDate) -> int:
    """
    Convert a 1‑based CosmicDate to total seconds since the start of the cosmic calendar.
    """
    return (
        (date.year - 1) * SECS_IN_YEAR +
        (date.month - 1) * (SECS_IN_DAY * MONTH) +   # month has MONTH days
        (date.day - 1) * SECS_IN_DAY +
        (date.hour - 1) * 3600 +
        (date.minute - 1) * 60 +
        (date.second - 1)                             # second is also 1‑based
    )

def seconds_to_cosmic_date(seconds: int) -> CosmicDate:
    """
    Convert total seconds since the start of the cosmic calendar back to a 1‑based CosmicDate.
    Wraps around after one Great Year.
    """
    total_secs_in_great_year = SECS_IN_YEAR * GREAT_YEAR
    seconds %= total_secs_in_great_year

    year = seconds // SECS_IN_YEAR + 1
    rem = seconds % SECS_IN_YEAR

    month_secs = SECS_IN_DAY * MONTH
    month = rem // month_secs + 1
    rem %= month_secs

    day = rem // SECS_IN_DAY + 1
    rem %= SECS_IN_DAY

    hour = rem // 3600 + 1
    rem %= 3600

    minute = rem // 60 + 1
    rem %= 60

    second = rem + 1                     # rem is 0‑59 → second becomes 1‑60

    return CosmicDate(year, month, day, hour, minute, second)

def iter_cosmic_dates(start_date: CosmicDate, end_seconds: int, step_seconds: int):
    current_seconds = cosmic_date_in_seconds(start_date)
    end_seconds_total = current_seconds + end_seconds
    while current_seconds < end_seconds_total:
        yield seconds_to_cosmic_date(current_seconds)
        current_seconds += step_seconds

def precession_angle(date: CosmicDate) -> float:
    return PRECESSION_ANGLE_SEC * cosmic_date_in_seconds(date)

def calc_date_values(date: CosmicDate):
    assert valid_date(date), "date not valid"
    sec = cosmic_date_in_seconds(date)
    return {"sec": sec, "precession_angle": PRECESSION_ANGLE_SEC * sec}







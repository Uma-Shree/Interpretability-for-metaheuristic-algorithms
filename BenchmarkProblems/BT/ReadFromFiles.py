import csv
import re
from builtins import str
from collections import defaultdict

from BenchmarkProblems.BT.RotaPattern import WorkDay, RotaPattern
from BenchmarkProblems.BT.Worker import Worker

"""There are 2 files we care about:
* rosterPatternDays.csv
* exployeeSkillsData.csv


The purpose of this script will be to convert it into rosters and employees"""


def get_dicts_from_RPD(roster_pattern_days_file_name: str) -> list[dict]:
    with open(roster_pattern_days_file_name) as file:
        field_names = ("SYNTH_RP_ID", "DAY_INDEX", "START_TIME", "END_TIME", "LUNCH_DURATION")
        reader = csv.DictReader(file, fieldnames=field_names)
        return [line for line in reader][1:]


def get_workday_from_RP_row(rp_row: dict):
    """Example: RP_1,4,800,1610,40 -> Workday(working=True, start_time=800, end_time=1610)
                RP_1,5,  0,   0,40 -> Workday(working=False)"""
    start_time = int(rp_row["START_TIME"])
    end_time = int(rp_row["END_TIME"])

    if (start_time == 0) and (end_time == 0):
        return WorkDay.not_working()
    else:
        return WorkDay.working_day(start_time, end_time)


def make_roster_patterns_from_RPD(rpd_dicts: list[dict]) -> dict[str, RotaPattern]:
    main_dict = defaultdict(dict)
    for row in rpd_dicts:
        workday = get_workday_from_RP_row(row)
        rp_code = row["SYNTH_RP_ID"]
        day_index = int(row["DAY_INDEX"])
        main_dict[rp_code][day_index] = workday

    def convert_dict_for_rp_code(rp_code: str) -> RotaPattern:
        entries = main_dict[rp_code]
        days = [entries[index] for index in range(len(entries))]  # entries is a dictionary...
        # weeks of length 7 because the original dataset does not seem to care...
        return RotaPattern(workweek_length=7, days=days)

    return {key: convert_dict_for_rp_code(key)
            for key in main_dict}


def get_skills_dict(employee_skills_data_file: str) -> dict:


    with open(employee_skills_data_file, "r") as file:
        field_names = ("EMPLOYEE_ID","SKILL_ID","SKILL_PREFERENCE")
        reader = csv.DictReader(file, fieldnames=field_names)
        rows = [line for line in reader][1:]

    result = defaultdict(set)
    for row in rows:
        result[row["EMPLOYEE_ID"]].add(row["SKILL_ID"])

    return result




def get_dicts_from_ED(employee_data_file_name: str) -> list[dict]:
    with open(employee_data_file_name, "r") as file:
        field_names = (
        "ID", "DOMAIN", "ROSTER_PATTERN_ID", "PREFERENCE_1", "PREFERENCE_2", "PREFERENCE_3", "PREFERENCE_4", "PREFERENCE_5")
        reader = csv.DictReader(file, fieldnames=field_names)
        employee_dicts = [line for line in reader][1:]

    return employee_dicts





def make_employees_from_ED(employee_rows: list[dict], skills_dict: dict, rotas: dict[str, RotaPattern], names: list[str]):
    # names_available = list(names)

    def decode_employee(employee_row: dict):
        employee_code: str = employee_row["ID"]
        employee_rotas = []
        for i in range(1, 5):
            rp_code = employee_row[f"PREFERENCE_{i}"]
            if rp_code == "":
                break
            else:
                employee_rotas.append(rotas[rp_code])

        employee_index = int(re.findall("\d+", employee_code)[0])
        name = names[employee_index]
        #names_available.remove(name)

        return Worker(available_skills=skills_dict[employee_code],
                      worker_id=employee_code,
                      available_rotas=employee_rotas,
                      name=name)

    return [decode_employee(row) for row in employee_rows]





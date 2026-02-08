import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, List


class DataFrameHandler(ABC):
    def __init__(self, successor=None):
        self._successor = successor

    def set_successor(self, successor):
        self._successor = successor
        return successor

    @abstractmethod
    def handle(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass

class BaseHandler(DataFrameHandler):
    def handle(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self._successor:
            return self._successor.handle(dataframe)
        return dataframe

class LoaderCSV(BaseHandler):
    def handle(self, csv_path: str) -> pd.DataFrame:
        hh_df = pd.read_csv(csv_path, index_col=0)
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class RemoveUnusedColumns(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        features = ['Авто', 'Последенее/нынешнее место работы', 'Ищет работу на должность:', 'Обновление резюме']
        hh_df = hh_df.drop(columns=features)
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessAgeSex(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        sex_age_series = hh_df["Пол, возраст"].to_list()
        drops = []
        i = 0
        sex = []
        age = []
        dirthday = []
        for entry in sex_age_series:
            data = entry.split(",")
            data = [item.replace(" ", "") for item in data]
            if len(data) < 2:
                drops.append(i)
                i = i + 1
                continue
            sex.append(data[0])
            age.append(data[1][:2])
            i = i + 1
        hh_df = hh_df.drop(hh_df.index[drops])
        hh_df = hh_df.reset_index(drop=True)
        age = [int(a) for a in age]
        sex_correct_list = []
        for item in sex:
            if item == "Female" or item == "Женщина":
                sex_correct_list.append("0")
            else:
                sex_correct_list.append("1")
        age = [int(i) for i in age]
        hh_df["Пол"] = sex_correct_list
        hh_df["Возраст"] = age
        hh_df = hh_df.drop(columns=["Пол, возраст"])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessEmployment(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        full = []
        partly = []
        volunteer = []
        w_place = []
        proj = []
        for item in hh_df["Занятость"].to_list():
            if "полная" in item.lower() or "full" in item.lower():
                full.append(1)
            else:
                full.append(0)
            if "частичная" in item.lower() or "part" in item.lower():
                partly.append(1)
            else:
                partly.append(0)
            if "волонтер" in item.lower() or "volunteer" in item.lower():
                volunteer.append(1)
            else:
                volunteer.append(0)
            if "проект" in item.lower() or "project" in item.lower():
                proj.append(1)
            else:
                proj.append(0)
            if "стаж" in item.lower() or "placement" in item.lower():
                w_place.append(1)
            else:
                w_place.append(0)
        hh_df["Полная занятость"] = full
        hh_df["Частичная занятость"] = partly
        hh_df["Проект"] = proj
        hh_df["Волонтерство"] = volunteer
        hh_df["Стажировка"] = w_place
        hh_df = hh_df.drop(columns=['Занятость'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessCity(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        relocation = []
        bt = []
        for item in hh_df['Город'].to_list():
            if not "не готов к переезду" in item.lower() and not "не готова к переезду" in item.lower() and not "not willing to relocate" in item.lower():
                relocation.append(1)
            else:
                relocation.append(0)
            if not "не готов к командировкам" in item.lower() and not "не готова к переезду" in item.lower() and not "not prepared for business trips" in item.lower():
                bt.append(1)
            else:
                bt.append(0)
        hh_df["Готовность к переезду"] = relocation
        hh_df["Готовность к командировкам"] = bt
        hh_df = hh_df.drop(columns=['Город'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessSchedule(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        full_day = []
        remote = []
        flexible = []
        rotation = []
        shift = []
        for item in hh_df["График"].to_list():
            if "полный" in item.lower() or "full" in item.lower():
                full_day.append(1)
            else:
                full_day.append(0)
            if "удал" in item.lower() or "remote" in item.lower():
                remote.append(1)
            else:
                remote.append(0)
            if "гибк" in item.lower() or "flexible" in item.lower():
                flexible.append(1)
            else:
                flexible.append(0)
            if "вахт" in item.lower() or "rotation" in item.lower():
                rotation.append(1)
            else:
                rotation.append(0)
            if "смен" in item.lower() or "shift" in item.lower():
                shift.append(1)
            else:
                shift.append(0)
        hh_df["Полный день"] = full_day
        hh_df["Удаленная работа"] = remote
        hh_df["Гибкий график"] = flexible
        hh_df["Вахтовый метод"] = rotation
        hh_df["Сменный график"] = shift
        hh_df = hh_df.drop(columns=['График'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessEducation(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        df_edu = []
        education_list = [item for item in hh_df["Образование и ВУЗ"].to_list()]
        drop_idxs = []
        idx = 0
        for item in education_list:
            if "высшее" in item.lower() or "higher" in item.lower():
                df_edu.append(4)
            elif "неоконченное" in item.lower() or "incomplete" in item.lower():
                df_edu.append(3)
            elif "специальное" in item.lower():
                df_edu.append(2)
            elif "среднее" in item.lower():
                df_edu.append(1)
            else:
                drop_idxs.append(idx)
            idx = idx + 1
        hh_df = hh_df.drop(hh_df.index[drop_idxs])
        hh_df = hh_df.reset_index(drop=True)
        hh_df["Образование"] = df_edu
        hh_df = hh_df.drop(columns=['Образование и ВУЗ'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessExperience(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        output = []
        drops = []
        idx = 0
        exp_list = [item.split("\n")[0] for item in hh_df['Опыт (двойное нажатие для полной версии)'].to_list()]
        for exp in exp_list:
            nums = list(map(int, re.findall(r'\d+', exp)))
            if len(nums) == 2:
                output.append(int(nums[0] * 12) + int(nums[1]))
            elif len(nums) == 1:
                if "мес" in exp or "month" in exp:
                    output.append(nums[0])
                else:
                    output.append(nums[0] * 12)
            else:
                drops.append(idx)
                idx = idx + 1
                continue
            idx = idx + 1 
        hh_df = hh_df.drop(hh_df.index[drops])
        hh_df = hh_df.reset_index(drop=True)
        hh_df["Опыт работы"] = output
        hh_df = hh_df.drop(columns=['Опыт (двойное нажатие для полной версии)'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessIncome(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        amount_list = [item.split(" ")[0].split("\xa0") for item in hh_df["ЗП"].to_list()]
        amount_list = [float("".join(it)) for it in amount_list]
        currency_list = [item.split(" ")[-1] for item in hh_df["ЗП"].to_list()]
        currency_dict = {
            "AZN": 45,
            "BYN": 27,
            "EUR": 91,
            "KGS": 0.88,
            "KZT": 0.16,
            "RUB": 1,
            "UAH": 1.79,
            "USD": 77,
            "бел.\xa0руб.": 26.81,
            "грн.": 1.79,
            "руб.": 1
        }
        output = []
        drops = []
        idx = 0
        for i in range(len(hh_df)):
            if currency_list[i] not in currency_dict.keys():
                drops.append(idx)
                idx = idx + 1
                continue
            else:
                output.append(amount_list[i] * currency_dict[currency_list[i]])
            idx = idx + 1
        hh_df = hh_df.drop(hh_df.index[drops])
        hh_df = hh_df.reset_index(drop=True)
        hh_df["Зарплата"] = output
        hh_df = hh_df.drop(columns=['ЗП'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ProcessEmployeeLevel(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        output = []
        exp_list = hh_df["Опыт работы"].to_list()
        job_titles = hh_df["Последеняя/нынешняя должность"].to_list()
        for i in range(len(hh_df)):
            if exp_list[i] >= 60 or "старш" in job_titles[i] or "ведущ" in job_titles[i] or "эксперт" in job_titles[i] or "директор" in job_titles[i] or "senior" in job_titles[i]:
                output.append(3)
            elif (exp_list[i] >= 24 and exp_list[i] < 60) or "middle" in job_titles[i]:
                output.append(2)
            else:
                output.append(1)
        hh_df["Уровень кандидата"] = output
        hh_df = hh_df.drop(columns=['Последеняя/нынешняя должность'])
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class SaveBarChart(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> pd.DataFrame:
        counts = hh_df['Уровень кандидата'].value_counts()
        levels = counts.index.to_list()
        plt_x = []
        for level in levels:
            if level == 1:
                plt_x.append("junior")
            elif level == 2:
                plt_x.append("middle")
            else:
               plt_x.append("senior") 
        plt.figure(figsize=(8,5))
        plt.bar(plt_x, counts.values.tolist(), color=['red', 'green', 'yellow'])
        plt.title("Распределение резюме по уровням специалиста")
        plt.xlabel("Уровень специалиста")
        plt.ylabel("Количество резюме")
        plt.yticks(np.arange(0, 50000, 5000))
        plt.grid(axis='y')
        plt.savefig('bar_chart.png')
        if self._successor:
            return self._successor.handle(hh_df)
        return hh_df

class ConvertToNumpy(BaseHandler):
    def handle(self, hh_df: pd.DataFrame) -> List[np.ndarray, np.ndarray]:
        y = hh_df["Зарплата"]
        X = hh_df = hh_df.drop(columns=['Зарплата'])
        np.save("X_data.npy", X)
        np.save("y_data.npy", y)
        if self._successor:
            return self._successor.handle([X, y])
        return [X, y]

class GetProcessedDF:
    def __init__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        
    def get_dataframe(self) -> pd.DataFrame:
        handler1 = LoaderCSV()
        handler2 = ProcessAgeSex()
        handler3 = ProcessEmployment()
        handler4 = ProcessExperience()
        handler5 = ProcessEducation()
        handler6 = ProcessIncome()
        handler7 = ProcessSchedule()
        handler8 = ProcessEmployeeLevel()
        handler9 = ProcessCity()
        handler10 = RemoveUnusedColumns()
        handler1.set_successor(handler2)
        handler2.set_successor(handler3)
        handler3.set_successor(handler4)
        handler4.set_successor(handler5)
        handler5.set_successor(handler6)
        handler6.set_successor(handler7)
        handler7.set_successor(handler8)
        handler8.set_successor(handler9)
        handler9.set_successor(handler10)
        hh_df = handler1.handle(self.path_to_csv)
        return hh_df


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Enter the path to hh.csv dataset")
        exit(0)

    hh_csv_file = sys.argv[1]

    handler1 = LoaderCSV()
    handler2 = ProcessAgeSex()
    handler3 = ProcessEmployment()
    handler4 = ProcessExperience()
    handler5 = ProcessEducation()
    handler6 = ProcessIncome()
    handler7 = ProcessSchedule()
    handler8 = ProcessEmployeeLevel()
    handler9 = ProcessCity()
    handler10 = RemoveUnusedColumns()
    handler11 = SaveBarChart()
    handler12 = ConvertToNumpy()

    handler1.set_successor(handler2)
    handler2.set_successor(handler3)
    handler3.set_successor(handler4)
    handler4.set_successor(handler5)
    handler5.set_successor(handler6)
    handler6.set_successor(handler7)
    handler7.set_successor(handler8)
    handler8.set_successor(handler9)
    handler9.set_successor(handler10)
    handler10.set_successor(handler11)
    handler11.set_successor(handler12)

    final_df = handler1.handle(hh_csv_file)

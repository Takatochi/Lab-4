import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

class DataStore:
    def __init__(self, filename):
        #конструктор даних 
        self.df = pd.read_csv(filename, index_col='Store ID ')
        #Перевірка даних, ввиід пеших 5 елментів 
        self.PrintDataSotre()
        # замінани назви зміної
        self.df = self.df.rename(columns={'Store_Area':'Stores','Store_Sales':'Sales'})
        # Скріпт вивода інформації
        self.PrintDataSotre()

    def correlation(self):
         # Розраховуємо кореляційну матрицю
        return self.df.corr()

    def min_correlation(self):
        # отримуємо кореляційну матрицю
        correlation = self.correlation()
        print("\ncorrelation :")
        print(correlation)

        # Знаходимо мінімальне значення кореляції
        min_correlation_column=self.correlation().idxmin()
        print("\nСтовпці з найнижчою кореляцією :")
        
        print(min_correlation_column)
        return min_correlation_column

    def PrintDataSotre(self):
        # Виводимо перші п'ять рядків даних
        print(self.df.head())

        self.min_correlation()

    def heatmap(self):
        # Візуалізуємо кореляцію між змінними `Stores` і `Sales` за допомогою діаграми теплової карти
        sns.heatmap(self.df.iloc[:,:4].corr(), annot=True, cmap="Blues")
        plt.title('\nCorrelation')
        plt.show()

    def scatter(self):
        # Візуалізуємо графік між змінними `Stores` і `Sales`
        self.y = self.df['Sales']
        # independent variable for x axis
        self.x = self.df['Stores']
        # Створення точкової діаграми між змінними `Stores` і `Sales`
        plt.figure(figsize=(20, 10), constrained_layout=True)
        # Мітки осей і збільшенням розміру шрифту
        plt.plot(self.x, self.y, 'o', markersize=15)
        # Мітки осей і збільшенням розміру шрифту
        plt.ylabel('Sales', fontsize=30)
        plt.xlabel('Stores', fontsize=30)
        # розмір шрифту  по осі x і y
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Відобразіть діаграму розсіювання
        plt.show()

    def regression(self):
        # Обчислюємо нахил і перетин Y-лінії
        self.m, self.b = np.polyfit(self.x, self.y, 1)
      
        # Виводимо нахил
        print('\nThe slope of line is {:.2f}.'.format(self.m))
        # Виводимо перетин
        print('The y-intercept is {:.2f}.'.format(self.b))
        # Виводимо рівняння найкращої лінії регресії
        print('The best fit simple linear regression line is {:.2f}x + {:.2f}.'.format(self.m, self.b))

        # Обчислюємо нахил і перетин Y-лінії та значення r-квадрат
        linregres = linregress(self.x, self.y)
        # Виводимо нахил, відрізок і значення r-квадрат
        print("\nНахил:", linregres.slope)
        print("відрізок:", linregres.intercept)
        print("R-квадрат:", linregres.rvalue)
        print('Найкраще підходить лінія простої лінійної регресії {:.2f}x + {:.2f}.'.format(linregres.slope, linregres.intercept))
    
    def centroid_whole_point(self):
        #Обчислюємо центроїд(центральну точку), як середню точку всіх точок у наборі даних. Він робить це, обчислюючи суму всіх точок і розділяючи її на кількість точок.
        return (self.df.sum()) / self.df.size
    
    def centroid(self):
        # Обчислюємо центроїд(центральну точку), як середню точку по осях x і y
        return self.y.mean(), self.x.mean()

    def plot_with_centroid(self):
        # Виводимо центроїд
        print("\nцентроїд середня всіх точок у наборі даних :")
        print(self.centroid_whole_point())

         # Виводимо центроїд
        print('\nЦентроїд для набору даних x по осях x = {:.2f} та y = {:.2f}.'.format(self.centroid()[1], self.centroid()[0]))

        # Побудова графіку розсіювання та лінії регресії з центроїдою
        plt.plot(self.x, self.y, 'o', markersize=14, label='Магазини')
        # Побудова графіку центроїди
        plt.plot(self.centroid()[1], self.centroid()[0], '*', markersize=30, color='r')
        # Побудова графіку лінії регресії
        plt.plot(self.x, self.m * self.x + self.b, '-', label='Проста лінійнаї регресії', linewidth=4)
        # Мітки осей і збільшенням розміру шрифту
        plt.ylabel('Продажі', fontsize=30)
        plt.xlabel('Кількість магазинів у районі', fontsize=30)
        # розмір шрифту  по осі x і y
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # Анотація для центроїди
        plt.annotate('Centroid', xy=(self.centroid()[1] - 0.1, self.centroid()[0] - 5), xytext=(self.centroid()[1] - 3, self.centroid()[0] - 20),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=30)
        plt.legend(loc='upper right', fontsize=20)
        plt.show()

    def predict_annual_sales(self, num_stores):
        """
        Прогнозує річний чистий обсяг продажів на основі кількості магазинів у районі.

        Args:
            num_stores: Кількість магазинів у районі.

        Returns:
            Прогнозований річний чистий обсяг продажів.
        """

        if num_stores < 1:
            raise ValueError("Щоб передбачити, у вас повинен бути принаймні 1 магазин в окрузірічний чистий обсяг продажів")
         
        # Прогнозування річного чистого обсягу продажів
        m, b =self.m, self.b
        annual_sales = m * num_stores + b
        return annual_sales
    

if __name__ == "__main__":
    #Ініціалізація класу 
    datastore = DataStore("stores-dist.csv")
    datastore.heatmap()
    datastore.scatter()
    datastore.regression()
    datastore.plot_with_centroid()

    # Введення кількості магазинів для прогнозу річних продажів
    num_stores_to_predict = 4 
    
     # Прогноз річних продажів
    predicted_annual_sales = datastore.predict_annual_sales(num_stores_to_predict)

    # Виведення прогнозу річних продажів
    print('\nПрогнозований річний обсяг продажів для {} магазинах: {:.2f}'.format(num_stores_to_predict, predicted_annual_sales))
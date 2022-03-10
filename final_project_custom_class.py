import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class PenguinsData:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.columns = self.df.columns
        
    def show_df(self):
        return self.df.head()

    def clean_species_column(self):
        self.df['Species'] = self.df['Species'].apply(lambda x: x.split()[0])

    def create_summary_table(self, group_cols, value_cols):
        try:
            return self.df.groupby(group_cols)[value_cols].mean()
        except KeyError as k:
            print("KeyError: caused due to invalid column name.")
            return k

    def show_unique_values(self, col_name):
        try:
            return self.df[col_name].value_counts()
        except KeyError as k:
            print("KeyError: check that col_name refers to valid columns in the data frame.")


    def species_island_prop(self):
        island_count = self.df.groupby('Island')['Species'].value_counts()
        island_prop = (100 * island_count / island_count.sum(level=0)).round(2)
        return pd.DataFrame(island_prop.reset_index(name = 'Percentage'))

    def box_plots(self, quant_var, hue=None):
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3))
            islands = {'Biscoe':0, 'Dream':1, 'Torgersen':2}
            df = self.df[self.df['Sex'] != '.']
            for island in islands:
                df_ = df[df['Island'] == island]
                ax_index = islands[island]
                sns.boxplot(x='Species', y=quant_var, hue=hue, data=df_, ax=axes[ax_index])
                axes[ax_index].set_title(island)                            
            
        except ValueError:
            print('ValueError: Perhaps, quant_var is not a valid column name.')

    def scatter_plot(self, x, y, cols=None, hue=None):
        df_ = self.df[self.df['Sex'] != '.']
        try:
            return sns.relplot(data=df_, x=x, y=y, col=cols, hue=hue)
        except ValueError as v:
            print('ValueError: Check that x, y, cols, and hue refer to valid columns in the data frame.')
            return v 


    def island_dist_bar_plot(self): 
        data = self.species_island_prop()
        return sns.barplot(data=data, x='Island', y='Percentage', hue='Species')

class PenguinClassifier():
    def __init__(self, PenguinsData, features, target, model):
        ## super().__init__()
        try:
            self.X = PenguinsData.df[features]
            self.y = PenguinsData.df[target]

            # remove null values
            is_na = self.X.isna().any(axis=1)
            self.X = self.X[~is_na]
            self.y = self.y[~is_na]
        except KeyError:
            print("KeyError: Check that features and target refer to valid column names.")
        
        self.model = model
        self.model_fit = None
        self.X_train, self.X_test = (None, None)
        self.y_train, self.y_test = (None, None)

    def show_X(self):
        return self.X.head()

    def show_y(self):
        return self.y[:6]
        
    def encode_data(self):
        try: 
            for column in ['Island', 'Sex']:
                if column in self.X.columns:
                    le = LabelEncoder()
                    self.X[column] = le.fit_transform(self.X[column])
                
            
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)   
        
        except ValueError:
            print('ValueError: Check that target is just a string and not a list of strings.')
            print('Label Encoder only accepts 1d arrays, so target cannot be a list of strings.')

    def split_data(self, test_size=0.20, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def fit_model(self):
        try:
            self.model_fit = self.model.fit(self.X_train, self.y_train) 
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)
            return {'model_instance': self.model, 'fit_model': self.model_fit, 'train_score':train_score, 'test_score':test_score}
        
        except AttributeError as a:
            print("AttributeError: Make sure that the model argument you passed is a valid sklearn model object.")
            return a
        

    def cv_scores(self, cv=5):
        try:
            return cross_val_score(self.model, self.X_train, self.y_train, cv=cv).mean()
        except TypeError as t:
            print("TypeError: The model should be an instance of an sklearn class.")
            print("This sklearn class should implement a fit method for cross validaiton to work.")
            return t
        
        except ValueError as v:
            print('Make sure that cv is an integer.')
            return v

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.model_fit.predict(self.X_test))

    def plot_confusion_matrix(self):
        y_pred = self.model_fit.predict(self.X_test)
        cf_matrix = confusion_matrix(self.y_test, y_pred)
        group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]        
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        ## print(len(group_counts))
        ## print(len(group_percentages))
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.array(labels).reshape(cf_matrix.shape)
        h_map = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        h_map.set_xticklabels(['Adelie', 'Chinstrap', 'Gentoo'])
        h_map.set_yticklabels(['Adelie', 'Chinstrap', 'Gentoo'])
        return h_map

def plot_decision_region(PenguinClassifier):
    x0 = PenguinClassifier.X_train['Culmen Depth (mm)']
    x1 = PenguinClassifier.X_train['Culmen Length (mm)']

    y = PenguinClassifier.y_train
    data = pd.DataFrame({
        'x0':x0, 
        'x1':x1, 
        'y':y
    })


    new_model = PenguinClassifier.model
    new_model.fit(PenguinClassifier.X_train[['Culmen Length (mm)', 'Culmen Depth (mm)']], PenguinClassifier.y_train)

    grid_x = np.linspace(x0.min(),x0.max(),501)
    grid_y = np.linspace(x1.min(),x1.max(),501)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    XX = xx.ravel()
    YY = yy.ravel()
    XY = pd.DataFrame({
        "Culmen Length (mm)" : YY,
        "Culmen Depth (mm)"  : XX
    })

    p = new_model.predict(XY)
    p = p.reshape(xx.shape)

    fig, ax = plt.subplots(1)
    
    # use contour plot to visualize the predictions
    ax.contourf(xx, yy, p, cmap = "tab10", alpha = 0.2, vmin = 0, vmax = 2)
    
    # plot the data
    sns.scatterplot(x='x0', y='x1', hue='y', data=data, palette = 'tab10', ax=ax)
    
    ax.set(xlabel = "Culmen Depth (mm)", 
           ylabel = "Culmen Length (mm)")

    


    


    




    
    
        
        



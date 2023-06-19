from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load datasets
customer_db = pd.read_csv('/Users/JGH/Python/Section Project 4/customer_db.csv')
drink_db = pd.read_csv('/Users/JGH/Python/Section Project 4/drink_db.csv')
order_db = pd.read_csv('/Users/JGH/Python/Section Project 4/order_db.csv')
drink_allergy = pd.read_csv('/Users/JGH/Python/Section Project 4/drink_allergy.csv')

# Data preprocessing
menu_count = order_db.groupby('drink_id')["id"].count().reset_index(drop=False)
menu_count.columns = ['id', 'sales']
drink = drink_db.merge(menu_count, on='id', how='inner')
drink = drink[drink.status=='Ice']
drink.reset_index(drop=True, inplace=True)
drink = drink.drop('status', axis=1)
drink['caffeine(mg)'] = drink['caffeine(mg)'].apply(lambda x: 'high' if x >= 53.25 else 'low')
drink['flavor'] = ['coffee','coffee','coffee','coffee','coffee',
                  'blend','coffee','coffee','coffee','coffee',
                  'coffee','coffee','blend','blend','blend',
                  'blend','blend','blend','blend','blend',
                  'blend','blend','blend', 'beverage','beverage',
                  'blend','beverage','beverage','beverage','beverage',
                  'beverage','beverage','beverage','beverage','beverage',
                  'beverage','beverage','beverage','beverage','beverage',
                  'beverage','beverage','beverage','beverage','beverage',
                  'beverage','blend','blend','blend','beverage',
                   'beverage','blend']
drink.columns = ['id', 'name', 'category', 'price', 'calories', 'sugars',
                'protein', 'sodium', 'saturated_fat', 'caffeine',
                'sales', 'flavor']
allergy_count = drink_allergy.groupby('name')['allergy'].count().reset_index()
drink = drink.merge(allergy_count, on='name', how='left')
drink['allergy'] = drink['allergy'].fillna(0)
ame_idx = drink[drink.name=='아이스 카페 아메리카노'].index.tolist()
drink = drink.drop(['id','name'], axis=1)
drink = drink.drop(ame_idx, axis=0)
drink.reset_index(drop=True, inplace=True)

# Splitting data into train and test sets
target = 'sales'
X = drink.drop(columns=target)
y = drink[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Scaling and Encoding
continuous_cols = ['price', 'calories', 'sugars', 'protein', 'sodium', 'saturated_fat', 'allergy']

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

ohe = OneHotEncoder()
cat_cols = ['category', 'caffeine', 'flavor']

X_train_encoded = ohe.fit_transform(X_train[cat_cols])
X_train_encoded_dense = X_train_encoded.toarray()
X_train_encoded_df = pd.DataFrame(X_train_encoded_dense, columns=ohe.get_feature_names(cat_cols))
X_train_encoded_df.index = X_train.index

X_test_encoded = ohe.transform(X_test[cat_cols])
X_test_encoded_dense = X_test_encoded.toarray()
X_test_encoded_df = pd.DataFrame(X_test_encoded_dense, columns=ohe.get_feature_names(cat_cols))
X_test_encoded_df.index = X_test.index

X_train = pd.concat([X_train[continuous_cols], X_train_encoded_df], axis=1)
X_test = pd.concat([X_test[continuous_cols], X_test_encoded_df], axis=1)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    price = float(request.form['price'])
    calories = float(request.form['calories'])
    sugars = float(request.form['sugars'])
    protein = float(request.form['protein'])
    sodium = float(request.form['sodium'])
    saturated_fat = float(request.form['saturated_fat'])
    allergy = int(request.form['allergy'])
    category = request.form['category']
    caffeine = request.form['caffeine']
    flavor = request.form['flavor']

    # Create an input data frame for prediction
    input_data = pd.DataFrame({
        'price': [price],
        'calories': [calories],
        'sugars': [sugars],
        'protein': [protein],
        'sodium': [sodium],
        'saturated_fat': [saturated_fat],
        'allergy': [allergy],
        'category': [category],
        'caffeine': [caffeine],
        'flavor': [flavor]
    })

    # Preprocess the input data
    input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])
    input_encoded = ohe.transform(input_data[cat_cols])
    input_encoded_dense = input_encoded.toarray()
    input_encoded_df = pd.DataFrame(input_encoded_dense, columns=ohe.get_feature_names(cat_cols))
    input_data = pd.concat([input_data[continuous_cols], input_encoded_df], axis=1)

    # Make a prediction
    prediction = model.predict(input_data)
    result = round(prediction[0])

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
import joblib

model=joblib.load("PlayTennis.pkl")
outlook=int(input("Please input the weather (0:Overcast, 1:Rainy, 2:Sunny): "))
temprature=int(input("How about the temprature? (0:Cool, 1:Hot, 2:Mild): "))
humidity=int(input("How about the humidity? (0:Humid, 1:Normal): "))
windy=int(input("Is it windy? (0:No, 1:Yes): "))

# Result
play=model.predict([[outlook,temprature,humidity,windy]])
if play:
    print("Then you can play outside")
else:
    print("Please consider to stay inside and do something else")
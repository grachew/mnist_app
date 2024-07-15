#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nest_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from threading import Thread
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array


model1 = load_model("model_mnist_1.keras")  # Инициализация модели
def process(image_file):

    # Открытие обрабатываемого файла
    image = load_img(BytesIO(image_file), target_size=(28, 28), color_mode='grayscale')
    array = img_to_array(image)
    
    array = array[np.newaxis, ...] # .[np.newaxis, ...] - добавление нового измерения (третьего)- (1,28,28)
    qq1 = array.reshape(1,784)
    li = []
    for i in range(len(qq1)):
        a = (255-qq1[i])/255
        li.append(a)
    qq2 = np.array(li)
    

    # Запуск предсказания
    prediction_array = model1.predict(qq2)

    # Возврат предсказания сети
    prdct = np.argmax(prediction_array)
    return  int(prdct)


nest_asyncio.apply() # Переиспользование событийного цикла в Jupyter Notebook
app = FastAPI() # Создание приложения FastAPI

# Маршрут
@app.post("/predict")
async def create_file(file: bytes = File(...)):
    return {"result": process(file)}

# Функция для запуска Uvicorn сервера в отдельном потоке
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8029)
    
# Запуск FastAPI приложения в отдельном потоке
thread = Thread(target=run_app)
thread.start()


# In[ ]:





# In[ ]:





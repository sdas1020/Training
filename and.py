from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging 

logging_str = "[%(levelname)s: %(asctime)s : %(module)s: %(lineno)] %(message)s"

logging.basicConfig(level=logging.INFO, format=logging_str)

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)
df
logging.info(f"This is the actual dataframe {df}")

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model, filename = "and.model")
save_plot(df, "and.png", model)

"""
try: 
    logging.info("Starting training")
    main()
except : Exception as e:
    logging.exception(e)
"""
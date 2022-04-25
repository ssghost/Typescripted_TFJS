import tensorflowjs as tfjs
from transformer import Transformer
import datetime

my_transformer = Transformer()
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
my_model_path = "./model_"+current_time
my_model = my_transformer.create_transformer().save(my_model_path)
tfjs.converters.save_keras_model(my_model, my_model_path+"_converted/")



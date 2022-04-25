import tensorflow as tf
import tensorflow_datasets as tfds

class Transformer:
    def __init__(self):
        self.model = None
        self.dataset = tfds.load('squad/v1.1', split='train', shuffle_files=True)

    class MultiHeadAttention(tf.keras.layers.Layer):
    
        def __init__(self, d_model, num_heads, name="multi_head_attention"):
            super(self.MultiHeadAttention, self).__init__(name=name)
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.query_dense = tf.keras.layers.Dense(units=d_model)
            self.key_dense = tf.keras.layers.Dense(units=d_model)
            self.value_dense = tf.keras.layers.Dense(units=d_model)

            self.dense = tf.keras.layers.Dense(units=d_model)

        def scaled_dot_product_attention(query, key, value, mask):
            matmul_qk = tf.matmul(query, key, transpose_b=True)
            depth = tf.cast(tf.shape(key)[-1], tf.float32)
            logits = matmul_qk / tf.math.sqrt(depth)
            if mask:
                logits += (mask * -1e9)
                attention_weights = tf.nn.softmax(logits, axis=-1)
            return tf.matmul(attention_weights, value)

        def split_heads(self, inputs, batch_size):
            inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(inputs, perm=[0, 2, 1, 3])

        def call(self, inputs):
            query, key, value, mask = inputs['query'], inputs['key'], inputs[
                'value'], inputs['mask']
            batch_size = tf.shape(query)[0]

            query = self.query_dense(query)
            key = self.key_dense(key)
            value = self.value_dense(value)

            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

            scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)

            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

            concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

            outputs = self.dense(concat_attention)

            return outputs

    class PositionalEncoding(tf.keras.layers.Layer):
    
        def __init__(self, position, d_model):
            super(self.PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_angles(self, position, i, d_model):
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model)
            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]
            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def encoder_layer(self,units, d_model, num_heads, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        attention = self.MultiHeadAttention(
            d_model, num_heads, name="attention")({
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': padding_mask
            })
        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def encoder(self, vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embeddings = self.PositionalEncoding(vocab_size, d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.encoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def decoder_layer(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = self.MultiHeadAttention(
            d_model, num_heads, name="attention_1")(inputs={
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': look_ahead_mask
            })
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = self.MultiHeadAttention(
            d_model, num_heads, name="attention_2")(inputs={
                'query': attention1,
                'key': enc_outputs,
                'value': enc_outputs,
                'mask': padding_mask
            })
        attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def decoder(self, vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        embeddings = self.PositionalEncoding(vocab_size, d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.decoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name='decoder_layer_{}'.format(i),
                )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def create_transformer(self, vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)

        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)

        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            )(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    def create_mask(self, inputs):
        embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        return embedding(inputs)

    def loss_function(self, y_true, y_pred, MAX_LENGTH=100):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred, MAX_LENGTH=100):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
        return accuracy

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
        def __init__(self, d_model, warmup_steps=4000):
            super(self.self.CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps**-1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def train(self):
        NUM_LAYERS = 2
        D_MODEL = 256
        NUM_HEADS = 8
        UNITS = 512
        DROPOUT = 0.1

        self.model = self.create_transformer(
                    vocab_size=50000,
                    num_layers=NUM_LAYERS,
                    units=UNITS,
                    d_model=D_MODEL,
                    num_heads=NUM_HEADS,
                    dropout=DROPOUT)

        learning_rate = self.CustomSchedule(D_MODEL)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])
        self.model.fit(self.dataset, epochs=20)
        return self.model

    def predict(self, sentence):
        import string
        sentence = sentence.apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
        sentence = sentence.apply(lambda wrd: ''.join(wrd))
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000)
        tokenizer.fit_on_texts(sentence)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        prediction = self.model.predict(le.fit_transform(sentence))
        predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
        return predicted_sentence

    
    
    
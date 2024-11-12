import tensorflow as tf


class BertClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert_model = bert_model

        self.fc = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def call(self, inputs):
        x = self.bert_model(inputs)
        x = x.pooler_output

        return self.fc(x)

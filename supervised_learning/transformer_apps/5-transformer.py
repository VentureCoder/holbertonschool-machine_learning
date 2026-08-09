#!/usr/bin/env python3
"""Transformer network for machine translation"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer

    Args:
        max_seq_len: integer representing the maximum sequence length
        dm: the model depth

    Returns:
        numpy.ndarray of shape (max_seq_len, dm) with the encoding vectors
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (index // 2)) / np.float32(dm))
    angle_rads = position * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention

    Args:
        Q: tensor with the query matrix
        K: tensor with the key matrix
        V: tensor with the value matrix
        mask: tensor that can be broadcast into the attention scores

    Returns:
        output, weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """Initializes the multi head attention layer

        Args:
            dm: the dimensionality of the model
            h: the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (h, depth) and transposes

        Args:
            x: tensor to split
            batch_size: the batch size

        Returns:
            the reshaped and transposed tensor
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Runs multi head attention

        Args:
            Q: tensor with the input to generate the query matrix
            K: tensor with the input to generate the key matrix
            V: tensor with the input to generate the value matrix
            mask: mask to apply, or None

        Returns:
            output, weights
        """
        batch_size = tf.shape(Q)[0]

        q = self.split_heads(self.Wq(Q), batch_size)
        k = self.split_heads(self.Wk(K), batch_size)
        v = self.split_heads(self.Wv(V), batch_size)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.dm))

        output = self.linear(attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Creates an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the encoder block

        Args:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Runs the encoder block

        Args:
            x: tensor with the input to the encoder block
            training: boolean, whether the model is training
            mask: mask to be applied for multi head attention

        Returns:
            tensor with the block's output
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        hidden = self.dense_hidden(out1)
        ffn_output = self.dense_output(hidden)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the decoder block

        Args:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask=None,
             padding_mask=None):
        """Runs the decoder block

        Args:
            x: tensor with the input to the decoder block
            encoder_output: tensor with the output of the encoder
            training: boolean, whether the model is training
            look_ahead_mask: mask for the first attention block
            padding_mask: mask for the second attention block

        Returns:
            tensor with the block's output
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        hidden = self.dense_hidden(out2)
        ffn_output = self.dense_output(hidden)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


class Encoder(tf.keras.layers.Layer):
    """Creates the encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the encoder

        Args:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: number of hidden units in the fully connected layer
            input_vocab: the size of the input vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Runs the encoder

        Args:
            x: tensor with the input to the encoder
            training: boolean, whether the model is training
            mask: mask to be applied for multi head attention

        Returns:
            tensor with the encoder output
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        pos_encoding = tf.cast(
            self.positional_encoding[:seq_len], tf.float32
        )
        x += pos_encoding
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Creates the decoder for a transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initializes the decoder

        Args:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: number of hidden units in the fully connected layer
            target_vocab: the size of the target vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask=None,
             padding_mask=None):
        """Runs the decoder

        Args:
            x: tensor with the input to the decoder
            encoder_output: tensor with the output of the encoder
            training: boolean, whether the model is training
            look_ahead_mask: mask for the first attention block
            padding_mask: mask for the second attention block

        Returns:
            tensor with the decoder output
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        pos_encoding = tf.cast(
            self.positional_encoding[:seq_len], tf.float32
        )
        x += pos_encoding
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Creates a transformer network"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the transformer

        Args:
            N: the number of blocks in the encoder and decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: number of hidden units in the fully connected layers
            input_vocab: the size of the input vocabulary
            target_vocab: the size of the target vocabulary
            max_seq_input: the maximum input sequence length
            max_seq_target: the maximum target sequence length
            drop_rate: the dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask=None,
             look_ahead_mask=None, decoder_mask=None):
        """Runs the transformer

        Args:
            inputs: tensor with the inputs
            target: tensor with the target
            training: boolean, whether the model is training
            encoder_mask: padding mask to be applied to the encoder
            look_ahead_mask: look ahead mask to be applied to the decoder
            decoder_mask: padding mask to be applied to the decoder

        Returns:
            tensor with the transformer output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask, decoder_mask
        )
        output = self.linear(decoder_output)

        return output

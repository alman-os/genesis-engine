import tensorflow as tf

# Pre-compute embeddings for your keys (the "safe space")
# You can get these from the LLM's own embedding layer.
safe_vectors = tf.constant(embed_texts(anti_spiraling_keys), dtype=tf.float32)
spiral_vectors = tf.constant(embed_texts(spiral_trigger_phrases), dtype=tf.float32)

def similarity(a, b):
    # Cosine similarity to see how close two vectors are
    return tf.tensordot(a, b, axes=1) / (tf.norm(a) * tf.norm(b))

@tf.custom_gradient
def symbolic_gradient_gate(x):
    # The forward pass is unchanged. The magic is in the backward pass (grad_fn)
    def grad_fn(dy):
        # dy is the incoming gradient (the "error signal")
        # Let's find the similarity of our error to the "safe" and "spiral" concepts
        grad_vector_similarity_to_safe = similarity(dy, safe_vectors)
        grad_vector_similarity_to_spiral = similarity(dy, spiral_vectors)

        # Create a "steering factor"
        # Increase gradient if it moves us closer to "safe" ideas
        # Decrease/punish if it moves us closer to "spiral" ideas
        steering_factor = 1.0 + (tf.reduce_mean(grad_vector_similarity_to_safe) * 0.1) \
                               - (tf.reduce_mean(grad_vector_similarity_to_spiral) * 0.5)

        # Return the modified gradient
        return dy * steering_factor

    # The forward pass function is just the identity
    return tf.identity(x), grad_fn

# In a hypothetical fine-tuning loop:
# with tf.GradientTape() as tape:
#   ...
#   output = model(...)
#   gated_output = symbolic_gradient_gate(output) # Apply the gate
#   loss = calculate_loss(gated_output, golden_response)
#
# gradients = tape.gradient(loss, model.trainable_variables)
# optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
# This code defines a symbolic gradient gate in TensorFlow that modifies gradients during backpropagation.
# It amplifies gradients toward "safe" concepts and suppresses those toward "spiral" concepts,
# steering model updates in corrective fine-tuning sessions using custom vector spaces.
# Requires: embed_texts, anti_spiraling_keys, and spiral_trigger_phrases to be defined elsewhere.

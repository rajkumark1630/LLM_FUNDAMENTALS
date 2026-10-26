Encoder-only models: BERT போன்ற models bidirectional attention பயன்படுத்தி input text-ன் இரு பக்க context-ஐயும் புரிந்துகொள்கின்றன; classification மற்றும் masked-token prediction-க்கு ஏற்றவை.
Decoder-only models: GPT-2 போன்ற models causal attention பயன்படுத்தி முந்தைய tokens அடிப்படையில் அடுத்த token-ஐ predict செய்கின்றன; text generation-க்கு ஏற்றவை.
Self-attention: ஒவ்வொரு token-க்கும் மற்ற tokens எவ்வளவு தொடர்புடையவை என்பதை attention weights மூலம் கணக்கிடுகிறது.
Multi-head attention: பல attention heads வெவ்வேறு token relationships-ஐ கற்றுக்கொள்ள உதவுகிறது.
Positional encoding: token order பற்றிய தகவலை model-க்கு வழங்குகிறது; இல்லையெனில் attention மட்டும் sequence order-ஐ நேரடியாகக் குறிக்காது.
Transformers vs RNNs: Transformers sequence positions-ஐ parallel-ஆக process செய்ய உதவுவதால் training வேகமாகவும் long-range relationships-ஐ கையாள்வது பயனுள்ளதாகவும் இருக்கலாம்; ஆனால் எல்லா tasks-லுமே RNN-களை விட சிறந்தவை என்று சொல்ல முடியாது.
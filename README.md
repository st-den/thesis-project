# Laundry Scan

This was a rather experimental PoC project I chose to do for my Bachelor of Software Engineering thesis. Basically, it's a cross-platform app that recognizes the laundry care symbols and shows their meaning.

The tech stack is the following: Python for a programming language, Kivy/KivyMD for a mobile development framework and Google Cloud Vision Auto ML (with Tensorflow inside) for the ML stuff. Multi-label classification was used as a machine learning approach.

The training set consists — roughly — of the available photos from a Kaggle [competition](https://www.kaggle.com/c/identification-care-symbols/discussion/96414) (1/4), Care Label Recognition [paper](https://cmp.felk.cvut.cz/~matas/papers/kralicek-2019-care_labels-icdar.pdf) dataset (1/4) and my own findings (2/4), totalling 200 pictures.

This turned out to be an unexpectedly hard problem to crack, but quite an interesting one, nevertheless. I profoundly enjoyed the challenge, and would gladly tell about the pitfalls of this topic. Feel free to contact me! :)

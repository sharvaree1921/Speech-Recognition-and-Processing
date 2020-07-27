# Speech-Recognition-and-Processing
## ERC_Wiki_Content
Speech Recognition is the process of converting spoken languages into text by computers. It is one of the sub-branch of the field of computational linguistics and computer science. Speech Recognition is what enables the machines and apps like Siri to understand what the person is saying and respond in the correct way. It is closely related to Natural Language Processing (NLP) and both may share some constructs. Speech Recognition only involves the conversion of an audio signal to corresponding text while NLP acts on this text to give meaningful data.

Speech Recognition is used in a lot of applications. The modern-day voice searches and voiced user interfaces like Amazon’s Alexa and Apple’s Siri relies on the speech recognition system for converting the spoken command to text. Further speech recognition can be used for voice recognition or speaker identification. 
(Speaker recognition is the identification of a person from characteristics of voices. It is used to answer the question "Who is speaking?")

### Background
Before we move on to recognizing the spoken languages, we must know how the words are composed and pronounced. In languages like English, the letters are pronounced in different ways depending on the word and even the context(the previous/next word can affect the pronunciation of the current word).
For pronunciation, we divide each word into syllables that contain a vowel with or without the surrounding consonants.

Vowels are syllabic speech sounds that are pronounced without any obstruction in the vocal tract. The vowel can be classified based on tongue height, tongue backness, and lip roundedness. Along with that vowel sounds can also be composed of two vowels, called a diphthong. Consonants are sounds that are articulated with a complete or partial closure of the vocal tract. The can be divided into various categories depending upon which part of the vocal tract is constricted as well as how much is constricted.
To distinguish between words, we use distinct units of sound called [phonemes](https://literarydevices.net/phoneme/). In English, we have 44 phonemes. In simple words, the phonemes are the sounds we produce when we speak a particular part of the word. For example, the word phoneme itself is composed of 6 phonemes: /ˈfəʊniːm/.
The phonemes may change a bit depending upon the surrounding phonemes. So to deal with the actual spoken language we have to actually deal with these modified phonemes which are called phones. So the main purpose of speech recognition can be said to identify these phones given an audio signal. 

### Signal Processing
The audio signal we receive is not much meaningful for speech recognition purposes and we have to use transformations to get meaningful data(often known as features) from it. For this, we use the [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) to convert the signal from the time domain to the frequency domain. But the data is of too much size that it is hard to handle and hence we take some of the most significant frequencies.

To apply the Fourier Transform, we need a sequence of data. But it is useless to process all the signals at once as the frequency changes over time and also sometimes it is not possible to get all the signals at once and hence we have to divide the signal into small durations(like 25 ms). We apply Fourier Transform for this small ‘frame’ and get the frequencies for that frame. This generates a spectrogram as shown below. This spectrogram depicts the amplitude(color-coded) of a given frequency(y-axis) in a given time frame(x-axis). 

![Speech Spectrum](https://github.com/sharvaree1921/Speech-Recognition-and-Processing/blob/master/Speech_spectrum.png)

Further, we apply some transformations to model the sound that we hear from what is produced(The sensitivity of the human ear varies non-linearly with frequency. We have to take into account this non-linearity as we are interested in what we hear than what is produced).


By applying these transformations, we get the features for the data. A better approach is to use [Mel-frequency cepstrum(MFC)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) and get the MFCC. We then use the first few coefficients as our features. These features are used by the model to find out the word that the audio signal represents. So now we have a few features per time frame that represents the whole data. This can be called as pre-processing the signal so that our model gets 'better quality' and 'concise' data so that the model is simpler and can perform more efficiently.

For more information about the transforms refer to [Common transforms in signal processing](https://en.wikibooks.org/wiki/Digital_Signal_Processing/Transforms). 

### Basics for the Model
To do speech recognition, given an audio signal we have to get the probability of that signal being a particular word/phone. We will assume that the word/phone with the highest probability is the given word/phone. So, the word/phone (W*) corresponding to the signal is given by: 
**W*=argmaxP(W|X)**
Here X is the feature vector calculated above using transformations, P(W|X) is the probability that the word represented by X is W and W* is the word/phone corresponding to feature X. It is difficult to make a model on this basis. So, we perform the following manipulation:

![Speech model](https://github.com/sharvaree1921/Speech-Recognition-and-Processing/blob/master/Speech_model.jpg)

In the second step, we apply the famous [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). In the last expression, we get W* in terms of two probabilities, P(X|W) which is the probability of observing the current observation given a word W, and P(W) which is the probability of the word being W. We calculate these two probabilities for all the possible words and select the one which has the maximum product of the two probabilities. 

It is easier to get the last expression (generative model). All modern speech recognition systems are based on this Basis. This is often referred to as the Naive Bayes Model. Since the last expression contains two different probabilities, the speech recognition system can be broken into parts to calculate the two probabilities and each part can be trained separately.

All Speech Recognition models can be broken into an acoustic model, a pronunciation lexicon model, and a language model though the modern speech recognition systems may combine these two into a single system.
The language model is about the likelihood of the word sequence. For example, “I watch a movie” will be more likely than “I you movie watch” or “I watch an apple”. It predicts the next word given the previous words. A pronunciation model can use tables to convert words to phones, or a corpus is already transcribed with phonemes. The acoustic model is about modeling a sequence of feature vectors given a sequence of phones instead of words. So in actuality, P(X|W) is composed of the pronunciation model and the acoustic model. These are modeled using the [Hidden Markov Model(HMM)](http://practicalcryptography.com/miscellaneous/machine-learning/hidden-markov-model-hmm-tutorial/) and [Gaussian Mixture Model(GMM)](http://practicalcryptography.com/miscellaneous/machine-learning/gaussian-mixture-model-tutorial/#the-expectation-maximisation-algorithm). 

Do give th read to HMMs and GMMs articles. They prove to be super useful.

![Overall model](https://github.com/sharvaree1921/Speech-Recognition-and-Processing/blob/master/Overall_model_for_speech_recognition.png)







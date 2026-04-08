import pickle


MODEL_PATH = "../models/naive_bayes_sentiment.pkl"


def loadModel(modelPath=MODEL_PATH):
    with open(modelPath, "rb") as file:
        modelPackage = pickle.load(file)
    return modelPackage


def predictSentiment(reviewText, modelPackage):
    pipeline = modelPackage["pipeline"]

    if isinstance(reviewText, str):
        reviewText = [reviewText]

    predictions = pipeline.predict(reviewText)
    probabilities = pipeline.predict_proba(reviewText)[:, 1]

    return predictions[0], probabilities[0]


if __name__ == "__main__":
    modelPackage = loadModel()
    print(f"Model: MultinomialNB Pipeline")
    print(f"Best params: {modelPackage['bestParameters']}")
    print(f"Test Accuracy: {modelPackage['testAccuracy']:.4f}")
    print(f"Test F1:       {modelPackage['testF1']:.4f}")
    print(f"Test AUC-ROC:  {modelPackage['testAucRoc']:.4f}")

    sampleReviews = [
        "This app is amazing, I love it! Best app ever.",
        "Terrible app, crashes all the time. Waste of space.",
        "It's okay, nothing special but gets the job done."
    ]

    print("\nSample predictions:")
    for review in sampleReviews:
        prediction, probability = predictSentiment(review, modelPackage)
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"  [{sentiment}] (p={probability:.3f}) {review[:60]}")

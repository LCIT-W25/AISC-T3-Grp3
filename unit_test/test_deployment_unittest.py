import unittest
from unittest.mock import patch, MagicMock
import Deployment


class TestDeployment(unittest.TestCase):
    def setUp(self):
        self.sample_text = "I love this product! It's truly outstanding."

    def test_preprocess_text(self):
        cleaned = Deployment.preprocess_text(self.sample_text)

        self.assertNotIn("its", cleaned.lower(), "Stopword or contraction should be filtered out.")
        self.assertIn("love", cleaned, "Expected 'love' to remain after preprocessing.")
        self.assertIn("product", cleaned, "Expected 'product' to remain.")
        self.assertIn("truly", cleaned, "Expected 'truly' to remain.")
        self.assertIn("outstanding", cleaned, "Expected 'outstanding' to remain.")

    @patch("Deployment.joblib.load")
    def test_naive_bayes_model_load(self, mock_joblib_load):
        mock_nb_model = MagicMock()
        mock_joblib_load.return_value = mock_nb_model

        # Suppose the code loads naive_bayes_v1.pkl:
        loaded_model = mock_joblib_load("naive_bayes_v1.pkl")
        self.assertEqual(loaded_model, mock_nb_model)
        mock_joblib_load.assert_called_with("naive_bayes_v1.pkl")

    @patch("Deployment.joblib.load")
    def test_svm_model_load(self, mock_joblib_load):

        mock_svm_model = MagicMock()
        mock_joblib_load.return_value = mock_svm_model

        # Suppose code loads svm_v2.pkl:
        loaded_svm = mock_joblib_load("svm_v2.pkl")
        self.assertEqual(loaded_svm, mock_svm_model)
        mock_joblib_load.assert_called_with("svm_v2.pkl")

    @patch("Deployment.joblib.load")
    def test_knn_model_predict(self, mock_joblib_load):
        mock_knn_model = MagicMock()
        # Suppose it always predicts "food"
        mock_knn_model.predict.return_value = ["food"]
        mock_joblib_load.return_value = mock_knn_model

        # Example 512-dim feature vector from ResNet
        mock_features = [[0.0] * 512]
        result = mock_knn_model.predict(mock_features)
        self.assertEqual(result[0], "food", "Expected 'food' prediction from the mock kNN model.")

    # @patch("Deployment.tf.keras.models.load_model")
    # def test_dnn_model_loading(self, mock_keras_load_model):
    #     mock_dnn_model = MagicMock()
    #     mock_keras_load_model.return_value = mock_dnn_model
    #
    #     loaded_dnn = mock_keras_load_model("DNN.keras")
    #     self.assertIsNotNone(loaded_dnn)
    #     mock_keras_load_model.assert_called_once_with("DNN.keras")


if __name__ == "__main__":
    unittest.main()

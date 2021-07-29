TEST_DIR=$1
CSV=$2

python predict_folder.py \
 --test-dir $TEST_DIR \
 --output $CSV \
 --models b7_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_last \
  b7_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_last \
  b7_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_last \
  b7_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_last \
  b7_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_last
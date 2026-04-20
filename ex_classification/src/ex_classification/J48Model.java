package ex_classification;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class J48Model {

    public static void run(String trainPath, String testPath,
                           TestModel mode, double percent, int folds) throws Exception {

        Instances train = new DataSource(trainPath).getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        J48 tree = new J48();

        // ⚙️ options
        tree.setConfidenceFactor(0.25f);
        tree.setMinNumObj(2);

        tree.buildClassifier(train);

        Evaluation eval;

        switch (mode) {

            case USE_TRAINING_SET:
                eval = new Evaluation(train);
                eval.evaluateModel(tree, train);
                break;

            case SUPPLIED_TEST_SET:
                Instances test = new DataSource(testPath).getDataSet();
                test.setClassIndex(test.numAttributes() - 1);

                eval = new Evaluation(train);
                eval.evaluateModel(tree, test);
                break;

            case PERCENT_SPLIT:
                train.randomize(new Random(1));

                int trainSize = (int) Math.round(train.numInstances() * percent / 100);
                int testSize = train.numInstances() - trainSize;

                Instances trainSplit = new Instances(train, 0, trainSize);
                Instances testSplit = new Instances(train, trainSize, testSize);

                tree.buildClassifier(trainSplit);

                eval = new Evaluation(trainSplit);
                eval.evaluateModel(tree, testSplit);
                break;

            case CROSS_VALIDATION:
                eval = new Evaluation(train);
                eval.crossValidateModel(tree, train, folds, new Random(1));
                break;

            default:
                throw new Exception("Invalid mode");
        }

        System.out.println(tree);
        System.out.println(eval.toSummaryString("\n=== J48 Result ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
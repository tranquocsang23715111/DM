package ex_classification;

import weka.classifiers.rules.OneR;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class OneRModel {

    public static void run(String trainPath, String testPath,
                           TestModel mode, double percent, int folds) throws Exception {

        Instances train = new DataSource(trainPath).getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        OneR model = new OneR();

        // ⚙️ options
        model.setMinBucketSize(6);

        model.buildClassifier(train);

        Evaluation eval;

        switch (mode) {

            case USE_TRAINING_SET:
                eval = new Evaluation(train);
                eval.evaluateModel(model, train);
                break;

            case SUPPLIED_TEST_SET:
                Instances test = new DataSource(testPath).getDataSet();
                test.setClassIndex(test.numAttributes() - 1);

                eval = new Evaluation(train);
                eval.evaluateModel(model, test);
                break;

            case PERCENT_SPLIT:
                train.randomize(new Random(1));

                int trainSize = (int) Math.round(train.numInstances() * percent / 100);
                int testSize = train.numInstances() - trainSize;

                Instances trainSplit = new Instances(train, 0, trainSize);
                Instances testSplit = new Instances(train, trainSize, testSize);

                model.buildClassifier(trainSplit);

                eval = new Evaluation(trainSplit);
                eval.evaluateModel(model, testSplit);
                break;

            case CROSS_VALIDATION:
                eval = new Evaluation(train);
                eval.crossValidateModel(model, train, folds, new Random(1));
                break;

            default:
                throw new Exception("Invalid mode");
        }

        System.out.println(model);
        System.out.println(eval.toSummaryString("\n=== OneR Result ===\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
package ex_classification;

public class MainApp {

    public static void main(String[] args) throws Exception {

        String trainFile = "C:\\Users\\Asus\\Kì 2 năm 3\\DataMining\\buy_computer_train.arff";
        String testFile = "C:\\Users\\Asus\\Kì 2 năm 3\\DataMining\\buy_computer_test.arff";

        //  chọn mode tại đây
        TestModel mode = TestModel.PERCENT_SPLIT;

        double percent = 66; // dùng cho PERCENT_SPLIT
        int folds = 10;      // dùng cho CROSS_VALIDATION

        System.out.println("===== J48 =====");
        J48Model.run(trainFile, testFile, mode, percent, folds);

        
        System.out.println("\n===== OneR =====");
        OneRModel.run(trainFile, testFile, mode, percent, folds);

        System.out.println("\n===== Naive Bayes =====");
        NaiveBayesModel.run(trainFile, testFile, mode, percent, folds);
    }
}
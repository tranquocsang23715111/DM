package ex_association;

public class MainApp {

    public static void main(String[] args) throws Exception {

        String filePath = "C:\\Program Files\\Weka-3-8-6\\data\\weather.nominal.arff";

        double minSup = 0.1;
        double minConf = 0.9;

        System.out.println("===== APRIORI =====");
        AprioriModel.run(filePath, minSup, minConf);

        System.out.println("\n===== FILTERED ASSOCIATOR =====");
        FilteredAssociatorModel.run(filePath, minSup, minConf);

        System.out.println("\n===== FP-GROWTH =====");
        FPGrowthModel.run(filePath, minSup, minConf);
    }
}
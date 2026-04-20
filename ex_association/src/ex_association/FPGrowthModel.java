package ex_association;

import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class FPGrowthModel {

    public static void run(String filePath, double minSup, double minConf) throws Exception {

        Instances data = new DataSource(filePath).getDataSet();

        // Discretize
        Discretize dis = new Discretize();
        dis.setInputFormat(data);
        Instances step1 = Filter.useFilter(data, dis);

        // Nominal -> Binary
        NominalToBinary nom2bin = new NominalToBinary();
        nom2bin.setInputFormat(step1);
        Instances step2 = Filter.useFilter(step1, nom2bin);

        // numeric -> nominal
        NumericToNominal num2nom = new NumericToNominal();
        num2nom.setOptions(new String[]{"-R", "first-last"});
        num2nom.setInputFormat(step2);
        Instances finalData = Filter.useFilter(step2, num2nom);

        // FP-Growth
        FPGrowth fp = new FPGrowth();
        fp.setLowerBoundMinSupport(minSup);
        fp.setMinMetric(minConf);
        fp.setNumRulesToFind(10);

        fp.buildAssociations(finalData);

        System.out.println("===== FP-GROWTH =====");
        System.out.println(fp);
    }
}
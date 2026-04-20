package ex_association;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AprioriModel {

    public static void run(String filePath, double minSup, double minConf) throws Exception {

        Instances data = new DataSource(filePath).getDataSet();

        Apriori apriori = new Apriori();

        apriori.setLowerBoundMinSupport(minSup);
        apriori.setMinMetric(minConf);
        apriori.setNumRules(20);

        apriori.buildAssociations(data);

        System.out.println(apriori);
    }
}
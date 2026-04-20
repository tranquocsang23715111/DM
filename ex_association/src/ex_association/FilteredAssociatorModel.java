package ex_association;

import weka.associations.Apriori;
import weka.associations.FilteredAssociator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class FilteredAssociatorModel {

    public static void run(String filePath, double minSup, double minConf) throws Exception {

        Instances data = new DataSource(filePath).getDataSet();

        // ví dụ: bỏ attribute cuối (class)
        Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.numAttributes()));

        Apriori apriori = new Apriori();
        apriori.setLowerBoundMinSupport(minSup);
        apriori.setMinMetric(minConf);

        FilteredAssociator fa = new FilteredAssociator();
        fa.setFilter(remove);
        fa.setAssociator(apriori);

        fa.buildAssociations(data);

        System.out.println(fa);
    }
}
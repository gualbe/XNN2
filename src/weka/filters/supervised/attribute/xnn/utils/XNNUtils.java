package weka.filters.supervised.attribute.xnn.utils;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.BitSet;

public class XNNUtils {

    public static Instance deepCopyInstance (Instance instance, boolean sameInstancesSet) {
        Instance copy;
        int i, n = instance.numAttributes();

        copy = new DenseInstance(instance);

        // No esta hecho el deep copy todavia.

        //for (i = 0; i < n; i++) {
        //    if (instance.attribute(i).isNominal())
        //}

        if (sameInstancesSet)
            copy.setDataset(instance.dataset());

        return copy;
    }

    public static Instances removeClass (Instances instances) {
        Remove remover = new Remove();
        Instances res = null;

        remover.setAttributeIndicesArray(new int[]{instances.classIndex()});

        try {
            remover.setInputFormat(instances);
            res = Filter.useFilter(instances, remover);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return res;
    }

    public static Instances removeAttribute (Instances instances, String attributeName) {
        RemoveByName remover = new RemoveByName();
        Instances res = null;

        remover.setExpression(attributeName);
        try {
            remover.setInputFormat(instances);
            res = Filter.useFilter(instances, remover);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return res;
    }

    public static Instances normalize (Instances instances) {
        Instances res = null;

        Filter filter = new Normalize();
        try {
            filter.setInputFormat(instances);
            res = Filter.useFilter(instances, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return res;
    }

    public static Instances standardize (Instances instances) {
        Instances res = null;

        Filter filter = new Standardize();
        try {
            filter.setInputFormat(instances);
            res = Filter.useFilter(instances, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return res;
    }

    public static Instances filterAttributes (Instances data, BitSet subset) {

        int i, j, n = data.numAttributes(), nsel;
        Instances dataCopy;
        Remove delTransform = new Remove();
        int[] featArray;

        delTransform.setInvertSelection(true);
        dataCopy = new Instances(data);

        // 1. Count attributes set in the BitSet
        nsel = 0;
        for (i = 0; i < n; i++) {
            if (subset.get(i)) {
                nsel++;
            }
        }

        // 2. Set up an array of attribute indexes for the filter (+1 for the class)
        j = 0;
        featArray = new int[nsel + 1];
        for (i = 0; i < n; i++) {
            if (subset.get(i)) {
                featArray[j] = i;
                j++;
            }
        }
        featArray[j] = data.classIndex();

        // 3. Produce the filtered dataset
        delTransform.setAttributeIndicesArray(featArray);
        try {
            delTransform.setInputFormat(dataCopy);
            dataCopy = Filter.useFilter(dataCopy, delTransform);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return dataCopy;
    }

    public static double distance (double [] a, double [] b) {
        int i, n = a.length;
        double res = 0;

        for (i = 0; i < n; i++)
            res += Math.pow(a[i] - b[i], 2);

        return Math.sqrt(res / n);
    }

    public static double distance(Instance instance1, Instance instance2) {
        int i, m = 0, n = instance1.numAttributes();
        double res = 0;

        for (i = 0; i < n; i++) {
            if (!instance1.isMissing(i) && !instance2.isMissing(i)) {
                if (instance1.attribute(i).isNominal())
                    res += instance1.value(i) == instance2.value(i) ? 0 : 1;
                else
                    res += Math.pow(instance1.value(i) - instance2.value(i), 2);
                m++;
            }
        }

        return Math.sqrt(res / (2 * m));
    }

    public static String vector2string (double[] v) {
        int i, n = v.length;
        String res = new String();

        res += "[";
        for (i = 0; i < n - 1; i++) {
            res += v[i] + ", ";
        }
        res += v[n - 1] + "]";
        return res;
    }


}

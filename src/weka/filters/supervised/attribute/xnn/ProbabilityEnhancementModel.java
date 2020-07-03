package weka.filters.supervised.attribute.xnn;

import weka.core.*;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighbor;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighborhood;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighborhoodTable;

import java.util.*;

public class ProbabilityEnhancementModel extends EnhancementModel implements OptionHandler {


    /**
     * Number of nearest neighbors for the XNNNeighborhoodTable construction.
     */
    protected int knn;

    /**
     * Default value for KNN.
     */
    public static final int KNN = 10;

    /**
     * Very small quantity to differentiate the probability of the actual class, in case of it is not the maximum one.
     */
    public static final double EPSILON = 0.000001;




    public ProbabilityEnhancementModel () {
        super();
        knn = KNN;
    }

    public ProbabilityEnhancementModel(boolean noNorm, boolean m_Debug, boolean toFileDebug) {
        super(noNorm, m_Debug, toFileDebug);
        knn = KNN;
    }

    @Override
    public Map<String, double[][]> generateEnhancementVectors() {

        Map<String, double[][]> res = Collections.synchronizedMap(new LinkedHashMap<String, double[][]>());
        XNNNeighborhoodTable table;
        double [][] distToBase;
        double [][] probs;
        double [][] adjusted_probs;

        // Step 1. Build the neighborhood table from instances.
        table = new XNNNeighborhoodTable(knn);
        table.build(instances);
        // System.out.println(table); // (to debug)

        // Step 2. Calculate probability vectors from the neighborhood table.
        probs = assessProbabilities(table);

        // Step 3. Calculate adjusted probabilities isolately (to debug).
        adjusted_probs = adjustProbabilities(probs);

        // Step 4. Calculate the distToBase values.
        distToBase = assessDistToBase(adjusted_probs);

        // Step 5. Pack the returned value.
        res.put("XNNProbUnadj", probs);
        res.put("XNNProbs", adjusted_probs);
        res.put("XNNDistToBase", distToBase);

        return res;

    }

    protected double [][] assessProbabilities (XNNNeighborhoodTable table) {

        double [][] res = new double[instances.numInstances()][numValoresClase];
        int i = 0;

        // Iterate for each instance to get its neighborhood and assess the probabilities.
        for (XNNNeighborhood row : table) {
            res[i] = assessProbabilities(row);
            i++;
        }

        return res;

    }

    public static double [] assessProbabilities (XNNNeighborhood nn) {
        return assessProbabilities_strategy1(nn);
    }

    private double [][] adjustProbabilities(double[][] probs) {
        int i, j, cl, n = instances.numInstances();
        double prob_base, prob_max, diff;
        double [][] res = new double[instances.numInstances()][numValoresClase];

        // Iterate for each instance.
        for (i = 0; i < n; i++) {
            cl = (int)instances.get(i).classValue();
            prob_base = probs[i][cl];

            // Get the maximum probability value for the instance.
            prob_max = 0;
            for (j = 0; j < numValoresClase; j++) {
                if (j != cl && probs[i][j] > prob_max)
                    prob_max = probs[i][j];
            }

            // If the prob_base is not the highest probability.
            if (prob_max >= prob_base) {
                res[i][cl] = prob_max + EPSILON; // set the new prob_base
                diff = res[i][cl] - prob_base; // assess the increment
                res[i][cl] /= 1 + diff; // normalize the new prob_base
                for (j = 0; j < numValoresClase; j++) { // normalize the rest of probabilities
                    if (j != cl)
                        res[i][j] = probs[i][j] / (1 + diff);
                }
            } else { // If the prob_base is the highest, then just copy the probabilities.
                for (j = 0; j < numValoresClase; j++) {
                    res[i][j] = probs[i][j];
                }
            }

        }

        return res;
    }

    protected static double [] assessProbabilities_strategy1 (XNNNeighborhood row) {

        int knn = row.getKnn(), numValoresClase = row.getNumValoresClase();
        double [] probs = new double[numValoresClase];
        int i; double sumDistances = 0, totalSum = 0;

        // Sum and max the distances of the neighborhood for this row.
        for (XNNNeighbor x : row) {
            sumDistances += x.distance;
        }

        // If all nearest neighbors are at distance zero.
        if (sumDistances == 0) sumDistances = 1;

        // Accumulate (sumDistances - distance) for each class.
        for (XNNNeighbor x : row) {
            probs[(int)x.theClass] += sumDistances - x.distance;
            totalSum += sumDistances - x.distance;
        }

        // Divide for each class to normalize probabilities.
        for (i = 0; i < numValoresClase; i++) {
            probs[i] /= totalSum;
            if (probs[i] < 0 || probs[i] > 1)
                System.err.println("\nprobs=" + probs[i] + "; sumDistances=" + sumDistances);
        }

        return probs;

    }

    protected double [][] assessDistToBase (double [][] probs) {
        double [][] res = new double[instances.numInstances()][1];
        int i, j, cl, n = instances.numInstances();
        double prob_base, prob_max, diff;

        // Iterate for each instance.
        for (i = 0; i < n; i++) {
            cl = (int) instances.get(i).classValue();
            prob_base = probs[i][cl];
            res[i][0] = 1 - (prob_base - 1 / numValoresClase) / (1 - 1 / numValoresClase);
        }

        return res;
    }

    @Override
    protected void init() {

    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {

        Instances result = new Instances(inputFormat, 0);
        int n = inputFormat.numClasses();

        // Set of attributes for probabilities.
        for (int i = 0; i < n; i++) {
            result.insertAttributeAt(new Attribute("XNNProbUnadj" + i), result.classIndex());
        }

        // Set of attributes for adjusted probabilities.
        for (int i = 0; i < n; i++) {
            result.insertAttributeAt(new Attribute("XNNProb" + i), result.classIndex());
        }

        // Attribute DistToBase.
        result.insertAttributeAt(new Attribute("XNNDistToBase"), result.classIndex());

        // Attribute SdForeign.
        // result.insertAttributeAt(new Attribute("SdForeign"), result.classIndex());

        return result;

    }

    public int getKnn() {
        return knn;
    }

    public void setKnn(int knn) {
        this.knn = knn;
    }

    public String knnTipText () {
        return "The number of neighbors used to enhance the class of instances.";
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setKnn(Integer.parseInt(knnString));
        } else {
            setKnn(KNN);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-K");
        result.add("" + this.knn);

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);
    }
}

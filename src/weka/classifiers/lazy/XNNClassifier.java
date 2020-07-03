package weka.classifiers.lazy;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighbor;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighborhood;

import java.util.*;

public class XNNClassifier extends AbstractClassifier implements OptionHandler {

    protected int knn;
    protected int numValoresClase;
    protected Instances instances;
    protected double [][] XNNProbs;

    /**
     * Default value for KNN.
     */
    public static final int KNN = 3;



    public XNNClassifier() {
        super();
        knn = KNN;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        EnhancementModel enhancer = new ProbabilityEnhancementModel();
        Map<String, double [][]> enhancementVectors;

        // Paso 1. Almacenar instancias de training.
        this.instances = data;
        this.numValoresClase = data.numClasses();

        // Paso 2. Generar los vectores de extensión usando el modelo XNN.
        enhancer.setInstances(data);
        enhancementVectors = enhancer.generateEnhancementVectors();

        // Paso 3. Obtener los vectores XNNProb de cada instancia.
        XNNProbs = enhancementVectors.get("XNNProbs");

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double [] distribution;
        double posMax = 0;
        int i;

        // Obtener el maximo del vector de probabilidades.
        distribution = distributionForInstance(instance);
        for (i = 0; i < numValoresClase; i++)
            if (distribution[i] > distribution[(int)posMax])
                posMax = i;

        return posMax;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        int i;
        XNNNeighborhood neighborhood;
        double [] res = new double[numValoresClase];

        // Paso 1. Obtener el vecindario de la instancia de test.
        neighborhood = new XNNNeighborhood(instance, 0, knn);
        neighborhood.processNeighbors(instances);

        // Paso 2. Obtener el vector de probabilidad promedio segun
        // vectores XNNProb de los vecinos mas cercanos del test.
        for (XNNNeighbor nn : neighborhood) {
            for (i = 0; i < numValoresClase; i++)
                res[i] += XNNProbs[nn.index][i];
        }
        for (i = 0; i < numValoresClase; i++)
            res[i] /= knn;

        return res;
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
        return super.listOptions();
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

    @Override
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }

}

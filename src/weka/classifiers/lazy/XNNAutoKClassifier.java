package weka.classifiers.lazy;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighbor;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighborhood;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;

public class XNNAutoKClassifier extends AbstractClassifier implements OptionHandler {

    protected int knnMin, knnMax;
    protected int numValoresClase;
    protected Instances instances;
    protected double [][] XNNProbs;
    protected double [][] distToBase;

    public static final int KNN_MIN = 1;
    public static final int KNN_MAX = 10;



    public XNNAutoKClassifier() {
        super();
        knnMin = KNN_MIN;
        knnMax = KNN_MAX;
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
        distToBase = enhancementVectors.get("XNNDistToBase");

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

        final int K_BASE = 1;
        int i; double k_optim = 0;
        XNNNeighborhood neighborhood;
        double [] res = new double[numValoresClase];

        // Paso 1. Obtener el primer vecindario de la instancia de test (con un valor de K base; p.ej. K=1).
        neighborhood = new XNNNeighborhood(instance, 0, K_BASE); // Podria configurarse este numero de vecinos.
        neighborhood.processNeighbors(instances);

        // Paso 2. Obtener el valor optimo de K para la instancia de test. // TODO : Continuar por aqui!!
        for (XNNNeighbor nn : neighborhood) {
            k_optim += this.distToBase[nn.index][0];
        }
        k_optim /= K_BASE;
        k_optim = k_optim * (KNN_MAX - KNN_MIN) + KNN_MIN;

        // Paso 3. Obtener el vecindario de la instancia de test segun K optimo.
        neighborhood = new XNNNeighborhood(instance, 0, (int)k_optim);
        neighborhood.processNeighbors(instances);

        // Paso 4. Obtener el vector de probabilidad promedio segun
        // vectores XNNProb de los vecinos mas cercanos del test.
        for (XNNNeighbor nn : neighborhood) {
            for (i = 0; i < numValoresClase; i++)
                res[i] += this.XNNProbs[nn.index][i];
        }
        for (i = 0; i < numValoresClase; i++)
            res[i] /= (int)k_optim;

        return res;
    }

    public int getKnnMin() {
        return knnMin;
    }

    public void setKnnMin(int knnMin) {
        this.knnMin = knnMin;
    }

    public int getKnnMax() {
        return knnMax;
    }

    public void setKnnMax(int knnMax) {
        this.knnMax = knnMax;
    }

    public String knnMinTipText () {
        return "The minimum number of neighbors considered to get the optimum k value.";
    }

    public String knnMaxTipText () {
        return "The maximum number of neighbors considered to get the optimum k value.";
    }

    @Override
    public Enumeration<Option> listOptions() {
        return super.listOptions();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String knnString = Utils.getOption('k', options);
        if (knnString.length() != 0) {
            setKnnMin(Integer.parseInt(knnString));
        } else {
            setKnnMin(KNN_MIN);
        }

        knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setKnnMax(Integer.parseInt(knnString));
        } else {
            setKnnMax(KNN_MAX);
        }

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-k");
        result.add("" + this.knnMin);
        result.add("-K");
        result.add("" + this.knnMax);

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);
    }

    @Override
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }

}

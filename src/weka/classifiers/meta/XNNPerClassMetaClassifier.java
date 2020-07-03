package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableMultipleClassifiersCombiner;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.trees.M5P;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.XNNClassEnhancer;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;

public class XNNPerClassMetaClassifier extends SingleClassifierEnhancer {

    protected Instances instances;
    protected Instances [] perClassInstances;
    protected Classifier [] perClassModels;
    protected int numValoresClase;


    public XNNPerClassMetaClassifier() {
        super();
        m_Classifier = new M5P();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        Instances enhanced_instances;
        Filter enhancer;
        int i, j;

        // Paso 1. Almacenar instancias de training.
        this.instances = data;
        this.numValoresClase = data.numClasses();

        // Paso 2. Agregar los vectores de extensión al training usando el modelo XNN.
        enhancer = new XNNClassEnhancer();
        enhancer.setInputFormat(this.instances);
        enhanced_instances = Filter.useFilter(this.instances, enhancer);

        // Paso 3. Para cada clase, generar un dataset con los atributos originales
        // y la clase original sustituida por la componente XNNProb correspondiente.
        perClassInstances = new Instances[numValoresClase];
        for (i = 0; i < numValoresClase; i++) {
            perClassInstances[i] = new Instances(enhanced_instances);
            perClassInstances[i] = XNNUtils.removeClass(perClassInstances[i]);
            perClassInstances[i] = XNNUtils.removeAttribute(perClassInstances[i], "XNNDistToBase");
            perClassInstances[i] = XNNUtils.removeAttribute(perClassInstances[i], "XNNProbUnadj" + i);
            for (j = 0; j < numValoresClase; j++) { // eliminar atributos XNNProb y XNNProbUnadj de las demas clases
                if (i != j) {
                    perClassInstances[i] = XNNUtils.removeAttribute(perClassInstances[i], "XNNProb" + j);
                    perClassInstances[i] = XNNUtils.removeAttribute(perClassInstances[i], "XNNProbUnadj" + j);
                }
            }
            perClassInstances[i].setClassIndex(perClassInstances[i].numAttributes()-1);
        }

        // Paso 4. Entrenar un modelo para cada dataset.
        perClassModels = new Classifier[numValoresClase];
        for (i = 0; i < numValoresClase; i++) {
            perClassModels[i] = m_Classifier.getClass().newInstance();
            perClassModels[i].buildClassifier(perClassInstances[i]);
        }

    }

    @Override
    protected String defaultClassifierString() {
        return "weka.classifiers.trees.M5P";
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
        double [] res = new double[numValoresClase];
        double sum = 0;
        int i;

        for (i = 0; i < numValoresClase; i++) {
            res[i] = perClassModels[i].classifyInstance(instance);
            sum += res[i];
        }

        for (i = 0; i < numValoresClase; i++) {
            res[i] /= sum;
        }

        return res;
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(4);

//        newVector.addElement(new Option(
//                "\tBase classifier (regressor).\n"
//                        + "\t(default M5P)",
//                "M", 1, "-M <classifier>"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
//        String temp = Utils.getOption('M', options);
//        this.baseClassifier = (Classifier) Class.forName(temp).newInstance();

        super.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<String>();

//        options.add("-M");
//        options.add("" + baseClassifier.getClass().getName());

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }


}

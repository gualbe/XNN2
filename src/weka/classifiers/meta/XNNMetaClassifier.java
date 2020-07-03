package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.XNNClassEnhancer;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.filters.supervised.attribute.xnn.utils.XNNNeighborhood;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

public class XNNMetaClassifier extends SingleClassifierEnhancer {

    protected Instances instances, enhanced_instances;
    protected Classifier model;
    protected int numValoresClase;
    protected XNNClassEnhancer enhancer;


    public XNNMetaClassifier() {
        super();
        m_Classifier = new J48();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        int i, j;

        // Paso 1. Almacenar instancias de training.
        this.instances = data;
        this.numValoresClase = data.numClasses();

        // Paso 2. Agregar los vectores de extensión al training usando el modelo XNN.
        enhancer = new XNNClassEnhancer();
        enhancer.setInputFormat(this.instances);
        enhanced_instances = Filter.useFilter(this.instances, enhancer);

        // Paso 3. Dejar solo componentes XNNProbUnadj.
        enhanced_instances = XNNUtils.removeAttribute(enhanced_instances, "XNNDistToBase");
        for (i = 0; i < numValoresClase; i++) {
            enhanced_instances = XNNUtils.removeAttribute(enhanced_instances, "XNNProb" + i);
        }

        // Paso 4. Entrenar un modelo para el dataset extendido.
        model = m_Classifier.getClass().newInstance();
        model.buildClassifier(enhanced_instances);

    }

    @Override
    protected String defaultClassifierString() {
        return "weka.classifiers.trees.J48";
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
        int i, n, knn = 10; // TODO : Hacer KNN una opcion del algoritmo.
        Instance enhancedInstance;

        // Paso 1. Obtener componentes XNNProbUnadj de la instancia de test.
        XNNNeighborhood row = new XNNNeighborhood(instance, 0, knn);
        row.processNeighbors(instances);
        double [] aux = ProbabilityEnhancementModel.assessProbabilities(row);

        // Paso 2. Preparar instancia de test extendida con las componentes XNNProbUnadj.
        enhancedInstance = new DenseInstance(instance);
        for (i = 0; i < numValoresClase; i++) {
            n = enhancedInstance.numAttributes() - 1;
            enhancedInstance.insertAttributeAt(n);
            enhancedInstance.setValue(n, aux[i]);
        }
        enhancedInstance.setDataset(enhanced_instances);

        // Paso 3. Usar el modelo entrenado para realizar la prediccion.
        res = model.distributionForInstance(enhancedInstance);

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

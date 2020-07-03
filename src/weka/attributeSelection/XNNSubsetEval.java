package weka.attributeSelection;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Map;


public class XNNSubsetEval extends ASEvaluation implements SubsetEvaluator, OptionHandler {

    protected int numInstances;
    protected Instances instances;
    protected EnhancementModel enhancer;


    @Override
    public void buildEvaluator(Instances data) throws Exception {

        // can evaluator handle data?
        getCapabilities().testWithFail(data);

        this.instances = data;
        this.numInstances = data.numInstances();
        this.enhancer = new ProbabilityEnhancementModel();

    }

    @Override
    public double evaluateSubset(BitSet subset) throws Exception {

        int i;
        double res;
        double [][] distToBase; // has always one column: [.][0]
        Instances processed_instances;
        Map<String, double [][]> enhancementVectors;

        // Paso 0. Comprobar si el subconjunto de atributos esta vacio.
        if (subset.length() == 0)
            return 0;

        // Paso 1. Filtrar atributos indicados en "subset".
        processed_instances = XNNUtils.filterAttributes(instances, subset);

        // Paso 2. Generar los vectores de extensión usando el modelo XNN.
        enhancer.setInstances(processed_instances);
        enhancementVectors = enhancer.generateEnhancementVectors();

        // Paso 3. Obtener la distancia al punto base nativo de cada instancia.
        distToBase = enhancementVectors.get("XNNDistToBase");

        // Paso 4. Calcular la distancia media al punto base nativo para el dataset filtrado por los atributos seleccionados.
        //         Se tiene en cuenta las ponderaciones por instancias en el calculo.
        res = 0;
        for (i = 0; i < numInstances; i++)
            res += distToBase[i][0] * processed_instances.get(i).weight();
        res /= processed_instances.sumOfWeights();

        // (Opcional) Salida de depuracion.
        // System.out.println("\nSubset: " + subset.toString() + " --> " + res + "  (" + (1 - res) + ")");

        return res;

    }

    /**
     * Returns the capabilities of this evaluator.
     *
     * @return the capabilities of this evaluator
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}

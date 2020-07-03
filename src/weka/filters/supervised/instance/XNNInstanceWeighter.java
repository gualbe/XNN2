package weka.filters.supervised.instance;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;

import java.util.Map;

public class XNNInstanceWeighter extends SimpleBatchFilter {

    @Override
    public String globalInfo() {
        return "A filter to weight the instances according to the distribution of classes within the neighborhood of each instance. "
                + "This filter is a part of the functionality of the XNN package."
                + "\n\nGualberto Asencio-Cortes, Ph.D. (guaasecor@upo.es)\n";
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return inputFormat;
    }

    @Override
    protected Instances process(Instances instances) throws Exception {

        EnhancementModel enhancer = new ProbabilityEnhancementModel();
        Map<String, double [][]> enhancementVectors;
        Instances res = new Instances(instances);
        double [][] distToBase; int i;

        // Paso 1. Generar los vectores de extensión usando el modelo XNN.
        enhancer.setInstances(instances);
        enhancementVectors = enhancer.generateEnhancementVectors();

        // Paso 2. Obtener los vectores XNNProb de cada instancia.
        distToBase = enhancementVectors.get("XNNDistToBase");

        // Paso 3. Asignar peso a cada instancia.
        i = 0;
        for (Instance inst : res) {
            inst.setWeight(1 - distToBase[i][0]);
            i++;
        }

        return res;

    }
}

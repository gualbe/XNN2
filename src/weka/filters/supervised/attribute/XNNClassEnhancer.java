package weka.filters.supervised.attribute;

import weka.filters.supervised.attribute.xnn.EnhancementModel;
import weka.filters.supervised.attribute.xnn.ProbabilityEnhancementModel;
import weka.gui.GenericObjectEditor;
import weka.filters.supervised.attribute.xnn.utils.XNNLog;
import weka.filters.supervised.attribute.xnn.utils.XNNLogSingleton;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.SimpleBatchFilter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class XNNClassEnhancer extends SimpleBatchFilter {

    /**
     * Las instancias de entrenamiento para la clasificación.
     */
    protected Instances training;

    /**
     * Tuplas de números reales asociadas a las instancias cuando existe
     * etiqueta nominal.-
     * <p>
     * Se trata de una n-tupla de números reales por cada instancia
     * que conforma la clase continua asociada a la etiqueta nominal.
     */
    protected Map<String, double[][]> enhancementVectors;

    /**
     * Modelo de mejora de clase.
     */
    protected EnhancementModel enhancementModel = new ProbabilityEnhancementModel();

    /**
     * Fichero de cálculos intermedios de depuración
     */
    private BufferedWriter fres;

    /**
     * Presentación de mensajes intermedios de depuración
     */
    boolean m_Debug;

    /**
     * Ventana para la presentación de mensajes intermedios en pantalla
     */
    XNNLog logs;

    /**
     * Destinar cálculos intermedios de depuración a un fichero
     */
    protected boolean toFileDebug;




    public XNNClassEnhancer() {
        init();
        setInfoBD();
    }

    protected Instances determineOutputFormat(Instances inputFormat) {
        return enhancementModel.determineOutputFormat(inputFormat);
    }

    protected Instances process(Instances inst) {
        int i, j, k, n, cl;
        double[] values;
        double [][] puntos_base;
        Instances result = new Instances(determineOutputFormat(inst), 0);
        Collection<double [][]> e;

        // Step 1. Build XNN artifacts.
        n = inst.numClasses();
        training = inst;
        if (m_Debug) {
            escribirLog("XNNClassEnhancer: generateEnhancementVectors...\n");
            escribirLog("Labels from 0 to " + (training.numClasses() - 1) + "\n");
        }
        enhancementModel.setInstances(inst);
        enhancementVectors = enhancementModel.generateEnhancementVectors();

        // Step 2. Store artifacts as new attributes in the dataset.
        for (i = 0; i < inst.numInstances(); i++) {

            // Step 2.1. Copy original instance values.
            values = new double[result.numAttributes()];
            k = 0;
            for (j = 0; j < inst.numAttributes(); j++) {
                if (j != inst.classIndex()) {
                    values[k] = inst.instance(i).value(j);
                    k++;
                }
            }

            // Step 2.2. Add new attribute values.
            e = enhancementVectors.values();
            for (double [][] temp : e) {
                for (j = 0; j < temp[i].length; j++) {
                    values[k++] = temp[i][j];
                }
            }

            // Step 2.3. Add class value at the end of the instance.
            values[result.numAttributes() - 1] = inst.instance(i).classValue();

            // Step 2.4. Add the new instance to the result returned by the filter.
            result.add(new DenseInstance(1, values));
        }

        return result;
    }

    public String globalInfo() {
        return "A filter to create a set of attributes with information of the neighborhood geometry of data. "
                + "This filter is a core functionality for the XNN package. Let suppose a nominal class with n values. "
                + "New generated attributes depend on the selected enhancement model."
                + "\n\nGualberto Asencio-Cortes, Ph.D. (guaasecor@upo.es)\n";
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.disableAllClasses();
        result.disable(Capability.STRING_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    public EnhancementModel getEnhancementModel() {
        return enhancementModel;
    }

    public void setEnhancementModel(EnhancementModel enhancementModel) {
        this.enhancementModel = enhancementModel;
    }

    public String enhancementModelTipText () {
        return "The enhancement model for the XNN ClassEnhancer filter";
    }

    public boolean getDebug() {
        return m_Debug;
    }

    public void setDebug(boolean newDebug) {
        m_Debug = newDebug;
    }

    public Map<String, double[][]> getEnhancementVectors() {
        return enhancementVectors;
    }

    /**
     * Returns an enumeration describing the available options
     *
     * @return an enumeration of all the available options
     */
    @Override
    public Enumeration<Option> listOptions() {
        Vector newVector = new Vector(11);

        newVector.addElement(new Option(
                "\tChoose a model for the class enhancement.\n"
                        + "\t(Default = ProbabilityEnhancementModel)",
                "M", 1, "-M <enhancement model>"));

        newVector.addAll(Collections.list(super.listOptions()));

        newVector.addElement(new Option(
                "",
                "", 0, "\nOptions specific to model "
                + getEnhancementModel().getClass().getName() + ":"));
        newVector.addAll(Collections.list(((OptionHandler) getEnhancementModel()).listOptions()));

        return newVector.elements();
    }

    /**
     * Parses a given list of options. Valid options are:<p>
     * <p>
     * -M <model and its arguments> <br>
     * Set the enhancement model to build
     * <p>
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String tmpStr, modelString = Utils.getOption('M', options);
        String [] tmpOptions = Utils.splitOptions(modelString);
        if (tmpOptions.length != 0) {
            tmpStr        = tmpOptions[0];
            tmpOptions[0] = "";
            this.setEnhancementModel(EnhancementModel.forName(tmpStr, tmpOptions));
        }
        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of IBk.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        result.add("-M");
        result.add("" + getEnhancementModel().getClass().getName() + " " + Utils.joinOptions(getEnhancementModel().getOptions()));

        Collections.addAll(result, super.getOptions());

        return (String[]) result.toArray(new String[result.size()]);
    }

    private void init(){
        GenericObjectEditor.registerEditor("weka.filters.supervised.attribute.xnn.EnhancementModel", "weka.gui.GenericObjectEditor");
        setEnhancementModel(new ProbabilityEnhancementModel());
    }

    public void setInfoBD() {
        if (m_Debug) {
            try {
                fres = new BufferedWriter(new FileWriter("XNNClassEnhancer.log"));
            } catch (IOException ioe) {
            }
        }
    }

    private void escribirLog(String log) {
        System.out.println("Escribir: "+log);
        try {
            if (toFileDebug) {
                if (fres == null) {
                    setInfoBD();
                }
                try {
                    fres.write(log);
                    fres.flush();
                } catch (IOException ioe) {
                }
            } else {
//                logs.txtPrincipal.append(log);
                XNNLogSingleton.getLogs().txtPrincipal.append(log);
            }
        }
        catch (Exception e){
            System.out.println("Excepción escribir log XNNFilter " + e.getMessage());
        }
    }

    public XNNLog getLogs() {
        return logs;
    }

    public void setLogs(XNNLog logs) {
        this.logs = logs;
    }

    public String logsTipText () {
        return "XNN Log";
    }

    public static void main(String[] args) {
        runFilter(new XNNClassEnhancer(), args);
    }

}
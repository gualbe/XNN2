package weka.filters.supervised.attribute.xnn;

import weka.core.*;

import weka.filters.supervised.attribute.xnn.utils.XNNLog;
import weka.filters.supervised.attribute.xnn.utils.XNNUtils;
import weka.filters.supervised.attribute.xnn.utils.XNNLogSingleton;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.Map;
import java.util.Vector;


public abstract class EnhancementModel implements Serializable, OptionHandler {

    /**
     * Instancias cuyos atributos numéricos están normalizados entre 0 y 1
     */
    protected Instances instances;

    /**
     * Clase nominal o numérica
     */
    protected boolean claseNominal;

    /**
     * Número de valores para la clase (ó 1 para clase numérica)
     */
    protected int numValoresClase;

    /**
     * Normalizar o no
     */
    protected boolean noNorm;

    /**
     * Presentación de mensajes intermedios de depuración
     */
    protected boolean m_Debug;

    /**
     * Destinar cálculos intermedios de depuración a un fichero
     */
    protected boolean toFileDebug;

    /**
     * Ventana para la presentación de mensajes intermedios en pantalla
     */
    XNNLog logs;

    /**
     * Fichero de cálculos intermedios de depuración
     */
    private BufferedWriter fres;





    public EnhancementModel() {
        init();
    }

    public EnhancementModel(boolean noNorm, boolean m_Debug, boolean toFileDebug) {
        this();
        this.noNorm = noNorm;
        this.m_Debug = m_Debug;
        this.toFileDebug = toFileDebug;
    }

    public static EnhancementModel forName(String name, String[] options) throws Exception {
        return (EnhancementModel) Utils.forName(EnhancementModel.class, name, options);
    }


    public abstract Map<String, double[][]> generateEnhancementVectors();

    protected abstract void init();

    public abstract Instances determineOutputFormat(Instances inputFormat);


    public void setInstances(Instances instances) {
        if (noNorm)
            this.instances = instances;
        else
            this.instances = XNNUtils.standardize(instances); // Normalize or standardize?
        this.numValoresClase = instances.numClasses();
        this.claseNominal = instances.classAttribute().isNominal();
    }


    protected void escribirLog(String log) {
        if (toFileDebug) {
            if (fres == null) {
                openXNNLogFile();
            }
            try {
                fres.write(log);
                fres.flush();
            } catch (IOException ioe) {
            }
        } else {
            XNNLogSingleton.getLogs().txtPrincipal.append(log);
        }
    }

    public void openXNNLogFile() {
//        String r = JOptionPane.showInputDialog("Valor del umbral");
//        umbralSeguridadAcierto = Double.parseDouble(r);
        if (m_Debug) {
            try {
                fres = new BufferedWriter(new FileWriter("XNN.log"));
            } catch (IOException ioe) {
            }
        }
    }



    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> result = Option.listOptionsForClassHierarchy(this.getClass(), EnhancementModel.class);

        return result.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, EnhancementModel.class);

        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<String>();

        for (String s : Option.getOptionsForHierarchy(this, EnhancementModel.class)) {
            result.add(s);
        }

        return result.toArray(new String[result.size()]);
    }

}

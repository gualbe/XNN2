/*
 * ckNNykSkNNlog.java
 *
 * Created on 8 de abril de 2003, 2:59
 */

package weka.filters.supervised.attribute.xnn.utils;

/**
 *
 * @author  Gualberto
 */
public class XNNLog extends javax.swing.JFrame {

    /** Creates new form ckNNykSkNNlog */
    public XNNLog() {
        initComponents();
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    private void initComponents() {//GEN-BEGIN:initComponents
        jscrllpanPrincipal = new javax.swing.JScrollPane();
        txtPrincipal = new javax.swing.JTextArea();
        
        setTitle("C\u00e1lculos intermedios para ckNN y kSkNN");
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                exitForm(evt);
            }
        });
        
        txtPrincipal.setColumns(60);
        txtPrincipal.setTabSize(3);
        txtPrincipal.setRows(30);
        txtPrincipal.setBorder(new javax.swing.border.EtchedBorder());
        jscrllpanPrincipal.setViewportView(txtPrincipal);
        
        getContentPane().add(jscrllpanPrincipal, java.awt.BorderLayout.CENTER);
        
        pack();
    }//GEN-END:initComponents

    /** Exit the Application */
    private void exitForm(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_exitForm
        setVisible(false);
        dispose();
    }//GEN-LAST:event_exitForm

    /**
    * @param args the command line arguments
    */
    public static void main(String args[]) {
        new XNNLog().show();
    }


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JScrollPane jscrllpanPrincipal;
    public javax.swing.JTextArea txtPrincipal;
    // End of variables declaration//GEN-END:variables

}

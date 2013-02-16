/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package megamandel2;

import java.awt.GridLayout;
import javax.swing.JFrame;

/**
 *
 * @author thorsten
 */
public class MegaMandel2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        JFrame f = new JFrame();
        f.setSize(200, 200);
        f.setLayout(new GridLayout());
        f.add(new NewJPanel());
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}

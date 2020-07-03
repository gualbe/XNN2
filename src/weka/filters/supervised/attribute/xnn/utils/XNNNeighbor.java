package weka.filters.supervised.attribute.xnn.utils;

import java.text.DecimalFormat;

public class XNNNeighbor implements Comparable<XNNNeighbor> {

    public int index;
    public double theClass;
    public double distance;


    public XNNNeighbor() {

    }

    public XNNNeighbor(int index, double theClass, double distance) {
        this.index = index;
        this.theClass = theClass;
        this.distance = distance;
    }

    @Override
    public int compareTo(XNNNeighbor o) {
        if (distance == o.distance)
            return 0;
        else if (distance < o.distance)
            return 1;
        else
            return -1;
    }

    @Override
    public String toString () {
        return "(" + index + " " + theClass + " " + new DecimalFormat("#0.000").format(distance) + ")";
    }

}

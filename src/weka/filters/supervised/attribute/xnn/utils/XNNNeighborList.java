package weka.filters.supervised.attribute.xnn.utils;

import weka.core.Instance;

/*
 * A class for a linked list to store the nearest k neighbours
 * to an instance. We use a list so that we can take care of
 * cases where multiple neighbours are the same distance away.
 * i.e. the minimum length of the list is k.
 */
public class XNNNeighborList {

    /**
     * The first node in the list
     */
    public XNNNeighborNode m_First;

    /**
     * The last node in the list
     */
    public XNNNeighborNode m_Last;

    /**
     * The number of nodes to attempt to maintain in the list
     */
    public int m_Length = 1;

    /**
     * Creates the neighborlist with a desired length
     *
     * @param length the length of list to attempt to maintain
     */
    public XNNNeighborList(int length) {

        m_Length = length;
    }

    /**
     * Gets whether the list is empty.
     *
     * @return true if so
     */
    public boolean isEmpty() {

        return (m_First == null);
    }

    /**
     * Gets the current length of the list.
     *
     * @return the current length of the list
     */
    public int currentLength() {

        int i = 0;
        XNNNeighborNode current = m_First;
        while (current != null) {
            i++;
            current = current.m_Next;
        }
        return i;
    }

    /**
     * Inserts an instance neighbor into the list, maintaining the list
     * sorted by distance.
     *
     * @param distance the distance to the instance
     * @param instance the neighboring instance
     */
    public void insertSorted(double distance, Instance instance, int index) {

        if (isEmpty()) {
            m_First = m_Last = new XNNNeighborNode(distance, instance, index);
        } else {
            XNNNeighborNode current = m_First;
            if (distance < m_First.m_Distance) {// Insert at head
                m_First = new XNNNeighborNode(distance, instance, index, m_First);
            } else { // Insert further down the list
                for (; (current.m_Next != null) &&
                        (current.m_Next.m_Distance < distance);
                     current = current.m_Next)
                    ;
                current.m_Next = new XNNNeighborNode(distance, instance, index,
                        current.m_Next);
                if (current.equals(m_Last)) {
                    m_Last = current.m_Next;
                }
            }

            // Trip down the list until we've got k list elements (or more if the
            // distance to the last elements is the same).
            int valcount = 0;
            for (current = m_First; current.m_Next != null;
                 current = current.m_Next) {
                valcount++;
                if ((valcount >= m_Length) && (current.m_Distance !=
                        current.m_Next.m_Distance)) {
                    m_Last = current;
                    current.m_Next = null;
                    break;
                }
            }
        }
    }

    /**
     * Prunes the list to contain the k nearest neighbors. If there are
     * multiple neighbors at the k'th distance, all will be kept.
     *
     * @param k the number of neighbors to keep in the list.
     */
    public void pruneToK(int k) {

        if (isEmpty()) {
            return;
        }
        if (k < 1) {
            k = 1;
        }
        int currentK = 0;
        double currentDist = m_First.m_Distance;
        XNNNeighborNode current = m_First;
        for (; current.m_Next != null; current = current.m_Next) {
            currentK++;
            currentDist = current.m_Distance;
            if ((currentK >= k) && (currentDist != current.m_Next.m_Distance)) {
                m_Last = current;
                current.m_Next = null;
                break;
            }
        }
    }

    /**
     * Prints out the contents of the neighborlist
     */
//    public void printList() {
//
//        if (isEmpty()) {
//            escribirLog("Lista vacía");
//        } else {
//            NeighborNode current = m_First;
//            while (current != null) {
//                escribirLog("Vecino: clase= " + current.m_Instance.classValue()
//                        + ", distancia= " + current.m_Distance
//                        + ", índice = " + current.index
//                        + ", valor= (" + current.m_Instance + ")\n");
//                current = current.m_Next;
//            }
//        }
//    }
}

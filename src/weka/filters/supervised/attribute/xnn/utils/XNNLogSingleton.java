package weka.filters.supervised.attribute.xnn.utils;

import java.util.HashMap;
import java.util.Map;

public class XNNLogSingleton {

    private static Map<Integer, XNNLog> logsMap = new HashMap<Integer, XNNLog>();

    public static XNNLog getLogs(Integer i){
        if (!logsMap.containsKey(i))
            logsMap.put(i, createNewLogs());
        return logsMap.get(i);
    }
    public static XNNLog getLogs(){
        return XNNLogSingleton.getLogs(-1);
    }

    private static XNNLog createNewLogs(){
        XNNLog logs = new XNNLog();
        logs.show();
        logs.txtPrincipal.setText("");
        return logs;
    }

}

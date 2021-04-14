using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Utils
{
    public static float get_distance(Transform src, Transform dest)
    {
        float dis = 0f;

        float AgentX = src.position.x;
        float AgentZ = src.position.z;

        float GoalX = dest.position.x;
        float GoalZ = dest.position.z;

        dis = Mathf.Sqrt((GoalX - AgentX) * (GoalX - AgentX) + (GoalZ - AgentZ) * (GoalZ - AgentZ));

        return dis;
    }
}
 

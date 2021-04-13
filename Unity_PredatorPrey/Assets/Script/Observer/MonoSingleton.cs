using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class MonoSingleton<T> : MonoBehaviour where T : MonoSingleton<T>
{
    protected static T instance = null;
    public static T Instance
    {
        get
        {
            if (null == instance)
            {
                instance = GameObject.FindObjectOfType(typeof(T)) as T;

                if (null == instance)
                {
                    Debug.Log(string.Format("{0}�� ���� instance�� ����"), instance);
                    return null;
                }
            }
            return instance;
        }
    }
}
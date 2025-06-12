using System.Collections.Generic;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UIElements;
using scener.ui;


namespace scener.ws {

public class WebSocketClient__ : MonoBehaviour
{
    //Singleton instance
    public static WebSocketClient__ Instance { get; private set; }

    private WebSocket ws;
    private int expectedImages = 0;
    private bool awaitingImages = false;
    private readonly List<Texture2D> receivedImages = new();
    private Terminal terminal;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    async void Start()
    {
        ws.OnOpen += () => Debug.Log("Connection opened!");
        ws.OnError += e => Debug.Log("Error: " + e);
        ws.OnClose += e => Debug.Log("Connection closed!");
        ws.OnMessage += (bytes) =>
        {
            string message = null;
            try
            {
                message = System.Text.Encoding.UTF8.GetString(bytes);

                if (awaitingImages && int.TryParse(message, out int imageCount))
                {
                    expectedImages = imageCount;
                    Debug.Log($"Expecting {expectedImages} image(s)");
                    return;
                }

                IncomingMessage parsed = JsonConvert.DeserializeObject<IncomingMessage>(message);

                if (parsed.action == Action.GenerateImage)
                {
                    awaitingImages = true;
                    receivedImages.Clear();
                    Debug.Log("Awaiting images...");
                    //terminal.AddMessageToChat("<b>[Agent]</b>: " + parsed.message);
                }
                else if (parsed.action == Action.ConvertedSpeech)
                {
                    //terminal.AddMessageToChat("<b>[You]</b>: " + parsed.message);
                    Debug.Log($"Received converted audio message: {parsed.message}");
                }
                else if (parsed.action == Action.ThinkingProcess)
                {
                    Debug.Log($"Thinking process: {parsed.message}");
                    //terminal.AddMessageToChat("<b>[Agent]</b>: " + parsed.message);
                }
                else
                {
                    Debug.Log($"Received message: {parsed.message}");
                    //terminal.AddMessageToChat("<b>[Agent]</b>: " + parsed.message);
                }
            }
            catch
            {
                if (awaitingImages)
                {
                    Texture2D tex = new(2, 2);
                    if (tex.LoadImage(bytes))
                    {
                        receivedImages.Add(tex);
                        Debug.Log($"Received image {receivedImages.Count} from {expectedImages}");

                        if (receivedImages.Count == expectedImages)
                        {
                            awaitingImages = false;
                            expectedImages = 0;
                            DisplayImages(receivedImages);
                        }
                    }
                    else
                    {
                        Debug.LogWarning("Failed to load image");
                    }
                }
                else
                {
                    Debug.LogWarning("Received unhandled binary message");
                }
            }
        };

        await ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        ws?.DispatchMessageQueue();
#endif
    }

    public async void OnApplicationQuit()
    {
        await ws.Close();
    }

    public async System.Threading.Tasks.Task SendTextMessage(string message)
    {
        if (ws != null && ws.State == WebSocketState.Open)
        {
            await ws.SendText(message);
            Debug.Log($"Sent message: {message}");
        }
        else
        {
            Debug.LogWarning("WebSocket is not connected.");
        }
    }

    public async System.Threading.Tasks.Task SendBytesMessage(byte[] message)
    {
        if (ws != null && ws.State == WebSocketState.Open)
        {
            await ws.Send(message);
            Debug.Log("Sent binary message");
        }
        else
        {
            Debug.LogWarning("WebSocket is not connected.");
        }
    }
}

}

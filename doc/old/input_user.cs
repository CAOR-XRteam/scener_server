using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UIElements;
using scener.input;


namespace scener.ws {

public class WebSocketUIChat : MonoBehaviour
{
    public static WebSocketUIChat Instance { get; private set; }
    readonly WebSocket websocket;
    TextField chatInput;
    Button chatSendButton;
    Button micButton;
    private AudioClip recordedClip;
    private const int maxRecordDuration = 60;
    private const int standardFreq = 44100;
    public string microphoneDevice;
    private bool isRecording = false;
    //private Terminal terminal;

    void Start(){
        //---------------------------

        // Get UIDocument root
        var root = GetComponent<UIDocument>().rootVisualElement;
        //terminal = gameObject.AddComponent<Terminal>();

        // Find the TextField
        chatInput = root.Q("terminal").Q<VisualElement>("box_input").Q<TextField>("text_input");
        chatSendButton = root.Q("terminal").Q<VisualElement>("box_input").Q<Button>("button_send");
        micButton = root.Q("terminal").Q<VisualElement>("box_input").Q<Button>("button_mic");

        // Register callback
        chatInput.RegisterCallback<KeyDownEvent>(OnChatInputKeyDown);
        chatSendButton.clicked += OnSendClicked;

        microphoneDevice = Microphone.devices[0];
        micButton.clicked += ToggleRecording;

        //---------------------------
    }

    async void OnChatInputKeyDown(KeyDownEvent evt)
    {
        if (evt.keyCode == KeyCode.Return || evt.keyCode == KeyCode.KeypadEnter)
        {
            string message = chatInput.value.Trim();
            if (!string.IsNullOrEmpty(message))
            {
                await WebSocketClient.instance.SendTextMessage(
                    JsonConvert.SerializeObject(
                        new OutgoingMessageMeta { command = Command.Chat, type = OutputType.Text }
                    )
                );
                await WebSocketClient.instance.SendTextMessage(message);
                //terminal.AddMessageToChat("<b>[You]</b>: " + message);
                chatInput.value = ""; // clear input
            }
        }
    }

    async void OnSendClicked()
    {
        string message = chatInput.value.Trim();
        if (!string.IsNullOrEmpty(message))
        {
            await WebSocketClient.instance.SendTextMessage(
                JsonConvert.SerializeObject(
                    new OutgoingMessageMeta { command = Command.Chat, type = OutputType.Text }
                )
            );
            await WebSocketClient.instance.SendTextMessage(message);
            //terminal.AddMessageToChat("<b>[You]</b>: " + message);
            chatInput.value = "";
        }
    }

    async void ToggleRecording()
    {
        if (!isRecording)
        {
            Debug.Log("Started recording...");
            Debug.Log($"Mic: {microphoneDevice}");
            recordedClip = Microphone.Start(
                microphoneDevice,
                false,
                maxRecordDuration,
                standardFreq
            );
            isRecording = true;
        }
        else
        {
            Debug.Log("Stopped recording...");
            int pos = Microphone.GetPosition(microphoneDevice);
            Microphone.End(microphoneDevice);
            isRecording = false;

            byte[] audioData = VoiceInput.ConvertToWav(pos, recordedClip);

            await WebSocketClient.instance.SendTextMessage(
                JsonConvert.SerializeObject(
                    new OutgoingMessageMeta { command = Command.Chat, type = OutputType.Audio }
                )
            );
            await WebSocketClient.instance.SendBytesMessage(audioData);
        }
    }

    void OnApplicationQuit()
    {
        WebSocketClient.instance.OnApplicationQuit();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
    }
}

}

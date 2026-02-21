using UnityEngine;
using UnityEngine.UI;

public class InfoPanel : MonoBehaviour
{
    [SerializeField] private float displayDistance = 1.5f;
    [SerializeField] private int fontSize = 36;

    private Canvas _canvas;
    private Text _titleText;
    private Text _bodyText;
    private Transform _cameraAnchor;

    private void Start()
    {
        // Find CenterEyeAnchor from OVRCameraRig
        var rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null)
            _cameraAnchor = rig.centerEyeAnchor;
        else
            _cameraAnchor = Camera.main != null ? Camera.main.transform : transform;

        CreatePanel();
        Hide();

        ControllerRaycast.OnCubeSelected += OnCubeSelected;
    }

    private void OnDestroy()
    {
        ControllerRaycast.OnCubeSelected -= OnCubeSelected;
    }

    private void CreatePanel()
    {
        // World-space canvas
        var canvasObj = new GameObject("InfoPanelCanvas");
        canvasObj.transform.SetParent(transform);
        _canvas = canvasObj.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;

        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        canvasObj.AddComponent<GraphicRaycaster>();

        var rt = _canvas.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(600, 400);
        rt.localScale = Vector3.one * 0.001f;

        // Background
        var bgObj = new GameObject("Background");
        bgObj.transform.SetParent(canvasObj.transform, false);
        var bgRect = bgObj.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        var bgImage = bgObj.AddComponent<Image>();
        bgImage.color = new Color(0f, 0f, 0f, 0.8f);

        // Title
        var titleObj = new GameObject("Title");
        titleObj.transform.SetParent(canvasObj.transform, false);
        _titleText = titleObj.AddComponent<Text>();
        _titleText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (_titleText.font == null)
            _titleText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        _titleText.fontSize = fontSize;
        _titleText.fontStyle = FontStyle.Bold;
        _titleText.color = Color.white;
        _titleText.alignment = TextAnchor.UpperCenter;
        var titleRect = _titleText.GetComponent<RectTransform>();
        titleRect.anchorMin = new Vector2(0, 0.7f);
        titleRect.anchorMax = new Vector2(1, 0.95f);
        titleRect.offsetMin = new Vector2(20, 0);
        titleRect.offsetMax = new Vector2(-20, 0);

        // Body
        var bodyObj = new GameObject("Body");
        bodyObj.transform.SetParent(canvasObj.transform, false);
        _bodyText = bodyObj.AddComponent<Text>();
        _bodyText.font = _titleText.font;
        _bodyText.fontSize = fontSize - 8;
        _bodyText.color = Color.white;
        _bodyText.alignment = TextAnchor.UpperLeft;
        var bodyRect = _bodyText.GetComponent<RectTransform>();
        bodyRect.anchorMin = new Vector2(0, 0.05f);
        bodyRect.anchorMax = new Vector2(1, 0.65f);
        bodyRect.offsetMin = new Vector2(30, 0);
        bodyRect.offsetMax = new Vector2(-30, 0);
    }

    private void OnCubeSelected(SelectableCube cube)
    {
        if (cube == null)
        {
            Hide();
            return;
        }
        Show(cube);
    }

    public void Show(SelectableCube cube)
    {
        _titleText.text = $"Cube #{cube.cubeId}";
        _bodyText.text = cube.cubeInfo;

        // Position panel in front of user's view
        var canvasTransform = _canvas.transform;
        canvasTransform.position = _cameraAnchor.position + _cameraAnchor.forward * displayDistance;
        canvasTransform.rotation = Quaternion.LookRotation(
            canvasTransform.position - _cameraAnchor.position
        );

        _canvas.gameObject.SetActive(true);
    }

    public void Hide()
    {
        if (_canvas != null)
            _canvas.gameObject.SetActive(false);
    }
}

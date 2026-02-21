using UnityEngine;

public class CubeSpawner : MonoBehaviour
{
    [SerializeField] private float minZ = 1.5f;
    [SerializeField] private float maxZ = 4.0f;
    [SerializeField] private float spreadX = 3.0f;
    [SerializeField] private float spreadY = 2.0f;
    [SerializeField] private float cubeScale = 0.3f;
    [SerializeField] private Material cubeMaterial;
    [SerializeField] private Material cubeHighlightMaterial;

    private void Start()
    {
        int count = Random.Range(2, 21);
        int cols = Mathf.CeilToInt(Mathf.Sqrt(count));
        int rows = Mathf.CeilToInt((float)count / cols);

        float spacingX = spreadX / Mathf.Max(cols - 1, 1);
        float spacingY = spreadY / Mathf.Max(rows - 1, 1);
        float startX = -spreadX / 2f;
        float startY = -spreadY / 2f + 1.2f; // offset up to eye level

        int id = 1;
        for (int r = 0; r < rows && id <= count; r++)
        {
            for (int c = 0; c < cols && id <= count; c++)
            {
                float x = startX + c * spacingX + Random.Range(-0.1f, 0.1f);
                float y = startY + r * spacingY + Random.Range(-0.1f, 0.1f);
                float z = Random.Range(minZ, maxZ);

                GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.transform.position = new Vector3(x, y, z);
                cube.transform.localScale = Vector3.one * cubeScale;
                cube.transform.rotation = Quaternion.Euler(
                    Random.Range(0f, 30f),
                    Random.Range(0f, 360f),
                    Random.Range(0f, 30f)
                );
                cube.name = $"Cube_{id}";

                if (cubeMaterial != null)
                    cube.GetComponent<MeshRenderer>().material = cubeMaterial;

                var selectable = cube.AddComponent<SelectableCube>();
                selectable.Initialize(id, cubeMaterial, cubeHighlightMaterial);

                id++;
            }
        }
    }
}

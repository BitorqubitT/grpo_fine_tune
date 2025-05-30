
#![allow(dead_code)]
/// Reasoning goes here
/// and can be multi-line
fn overlay_images(source_image: String, background_image: String, x_iterations: i32, y_iterations: i32) -> Vec<String> {
    let mut filenames = Vec::new();
    for x in 0..=x_iterations {
        for y in 0..=y_iterations {
            filenames.push(format!("output_{}_{}.jpg", x, y));
        }
    }
    filenames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_images() {
        let source_image = "source.png".to_string();
        let background_image = "background.png".to_string();
        let x_iterations = 5;
        let y_iterations = 3;

        let expected_filenames = vec![
            "output_0_0.jpg",
            "output_0_1.jpg",
            "output_0_2.jpg",
            "output_1_0.jpg",
            "output_1_1.jpg",
            "output_1_2.jpg",
            "output_2_0.jpg",
            "output_2_1.jpg",
            "output_2_2.jpg",
            "output_3_0.jpg",
            "output_3_1.jpg",
            "output_3_2.jpg",
            "output_4_0.jpg",
            "output_4_1.jpg",
            "output_4_2.jpg"
        ];

        assert_eq!(overlay_images(source_image, background_image, x_iterations, y_iterations), expected_filenames);
    }
}

fn main() {
    println!("Hello World");
}

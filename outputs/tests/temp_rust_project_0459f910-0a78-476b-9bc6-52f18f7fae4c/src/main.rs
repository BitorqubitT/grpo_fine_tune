
#![allow(dead_code)]
fn generate_rider_report(riders: Vec<i32>, template: &str) -> String {
    let mut report = String::new();
    for r_id in riders {
        report.push_str(&format!("{}{}", "Rider ID: ", r_id));
        report.push('\n');
    }
    report.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_rider_report() {
        assert_eq!(
            generate_rider_report(vec![5, 8, 1], "{ID}"),
            "ID:\nID: 5\nID: 8\nID: 1"
        );
        assert_eq!(
            generate_rider_report(vec![10, 20, 30, 40], "{}"),
            "ID: \nID: 10\nID: 20\nID: 30\nID: 40"
        );
    }
}

fn main() {
    println!("Hello World");
}
